from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

# 아래 builder 모듈은 모델의 구성 요소들을 만들기 위한 함수를 정의한다.
from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)

# FUSIONMODELS는 모델을 등록하기 위한 데코레이터이다.
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

# 현재 디렉토리에 있는 base.py 파일에서 Base3DFusionModel 클래스를 가져온다.
from .base import Base3DFusionModel

# ?
__all__ = ["BEVFusion"]

# FUSIONMODELS 데코레이터를 이용해 BEVFusion 클래스를 등록한다.
# 이 클래스는 Base3DFusionModel 클래스를 상속한다.
@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    # BEVFusion 클래스의 생성자
    def __init__(
            self,
            # 딕셔너리 형태의 파라미터들
            encoders: Dict[str, Any],
            fuser: Dict[str, Any],
            decoder: Dict[str, Any],
            heads: Dict[str, Any],
            # 키워드 인자들
            **kwargs,
    ) -> None:
        # 부모 클래스인 Base3DFusionModel를 호출하는 부분
        super().__init__()

        # nn.ModuleDict()의 인스턴스인 self.encoders를 생성, 여러 모듈을 저장하는 딕셔너리 형태의 컨테이너이다.
        self.encoders = nn.ModuleDict()

        # encoders 딕셔너리에서 camera 키 값이 존재하는지 확인한다.
        if encoders.get("camera") is not None:

            # camera 모듈을 nn.ModuleDict()로 생성하고, 그 안에 backbone, neck, vtransfrom 모듈을 생성해 저장한다.
            # 이 모듈은 build_backbone, build_neck, build_vtransform 함수를 통해 생성된다.
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )

        # encoders 딕셔너리에서 lidar 키 값이 존재하는지 확인한다.
        if encoders.get("lidar") is not None:

            # lidar 모듈 내 voxelize 키의 설정 값을 확인한다.
            # max_num_points가 0보다 큰지 확인해 하드 보클라이징 모듈인지, 다이나믹 스캐터링 모듈인지 판단 후
            # 이에 따라 voxelize_module을 생성한다.
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])

            # lidar 모듈을 nn.ModuleDict()로 생성하고, 그 안에 voxelize 모듈과 backbone 모듈을 저장한다.
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )

            # lidar 모듈 내 voxelize_reduce 키에 있는 값을 확인해 self.voxelize_reduce를 설정, 값이 없는 경우 True
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        # encoders 딕셔너리에서 radar 키 값이 존재하는지 확인한다.
        if encoders.get("radar") is not None:

            # radar 모듈 내 voxelize 키의 설정 값을 확인한다.
            # max_num_points가 0보다 큰지 확인해 하드 보클라이징 모듈인지, 다이나믹 스캐터링 모듈인지 판단 후
            # 이에 따라 voxelize_module을 생성한다.
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])

            # radar 모듈을 nn.ModuleDict()로 생성하고, 그 안에 voxelize 모듈과 backbone 모듈을 저장한다.
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )

            # radar 모듈 내 voxelize_reduce 키에 있는 값을 확인해 self.voxelize_reduce를 설정, 값이 없는 경우 True
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        # fuser 값이 None이 아닌 경우, build_fuser 함수를 통해 self.fuser를 생성한다.
        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        # decoder 모듈을 nn.ModuleDict()로 생성하고 그 안에 그 안에 backbone 모듈과 neck 모듈을 저장한다.
        # 이 모듈은 build_backbone, build_neck 함수를 통해 생성된다.
        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )

        # heads 딕셔너리 내 각 항목에 대해 None이 아닌 경우, build_head 함수를 통해 self.heads에 모듈을 생성하고 저장한다.
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        # 만약 kwargs 내에 loss_scale 키가 있다면, 해당 값을 사용해 self.loss_scale을 설정한다.
        # 아닌 경우 빈 딕셔너리를 생성하고, heads 내 각 항목에 대해 1.0을 값으로 설정해 self.loss_scale을 생성한다.
        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # depth loss을 사용하는지에 대해 판단하는 부분
        # camera 모듈 내 vtransform 키에 있는 type 값을 확인해 아래 [] 중 하나라면 self.use_depth_loss를 Ture로 설정한다.
        # If the camera's vtransform is a BEVDepth version, then we're using depth loss.
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in [
            'BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        # 가중치 초기화
        self.init_weights()

    # camera 모듈이 self.encoders 내 있다면, camera 모듈 내의 backbone 모듈의 가중치를 초기화 한다.
    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    # extract_camera_features : 추출된 피처를 처리하는 부분
    # 입력 데이터를 처리하고 카메라 피처을 추출하며, 변환과 depth 손실 값은 작업을 수행하는 역할을 한다.
    # 메서드는 입력 이미지 x의 차원을 조정, camera 모듈의 backbone과 neck을 거쳐 특성을 추출한다.
    # 그 후 self.encoders["camera"]["backbone"]을 호출해 피처 변환을 한다.
    # 이 변환은 카메라와 관련된 작업들을 수행, depth_loss와 gt_depth를 통해 depth 손실을 계산하고 활용한다.
    # 이후 변환된 특성 x를 반환한다.
    def extract_camera_features(
            self,
            x, # 입력 이미지 데이터
            points, # 포인트 클라우드 데이터, lidar
            radar_points, # 포인트 클라우드 데이터, radar
            camera2ego, # 변환 행렬
            lidar2ego, # 변환 행렬
            lidar2camera, # 변환 행렬
            lidar2image, # 변환 행렬
            camera_intrinsics, # 카메라 내부 파라미터
            camera2lidar, # 변환 행렬
            img_aug_matrix, # 보정 매트릭스
            lidar_aug_matrix, # 보정 매트릭스
            img_metas, # 이미지 메타 데이터
            gt_depths=None, # 실제 depth 데이터
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss,
            gt_depths=gt_depths,
        )
        return x

    # extract_features : 입력 데이터를 처리해 피처를 추출하는 부분
    # 다양한 센서에서 피처를 추출하고 처리하는 일반적인 기능을 수행한다.
    # x : 입력 데이터 (이미지, 포인트 클라우드 등)
    # sensor : 센서 유형 (카메라, 라이다, 레이더 등)
    # 메서드는 self.vocelize를 호출, 입력 데이터를 보클라이징 하고 그 결과인 feats, coords, sizes를 얻는다.
    # 이후 batch_size를 계산하고, 해당 값과 feats, coords, sizes를 이용해 sensor 모듈의 backbone을 통해 피처를 추출한다.
    # 이후 추출된 피처 x를 반환한다.
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # voxelize : 입력으로 받은 points와 sensor에 따라 데이터를 병합해 3D 공간을 3D 그리드로 변환하는 작업을 수행
    @torch.no_grad()  # 해당 함수 내 모든 연산을 추적하지 않도록 해서 연산 속도를 높인다.
    @force_fp32()  # 함수 내 연산을 모두 32비트 연산으로 강제적으로 처리한다.

    # voxelize 메서드는 입력 데이터를 보클라이징하여 특성, 좌표 및 사이즈 정보를 반환하는 부분
    # points : 입력 데이터 (포인트 클라우드 등)
    # sensor : 센서 유형 (카메라, 라이다, 레이더 등)
    # voxelize 메서드는 입력 데이터를 반복, self.encoders[sensor]["voxelize"]를 호출해 보클라이징 결과를 얻는다.
    # 보클라이징 결과는 voxelize 모듈을 통해 생성된다.
    # 만약 결과가 3개의 값으로 이루어져 있다면 이는 보클라이징 된 피처, 좌표 및 사이즈 정보이다.
    # 이 정보를 feats, coords, sizes 리스트에 추가하고 텐서로 변환해 저장한다.
    # 마지막으로 보클라이징 된 피처들을 합산하거나 사이즈에 따라 정규화하는 작업을 수행한 후 결과를 반환한다.
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    # forward 메서드는 신경망 모델을 통해 forward pass를 수행하는 부분
    @auto_fp16(apply_to=("img", "points"))
    def forward(
            self,
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            depths,
            radar=None,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        # 이미지가 리스트인 경우 예외 처리
        if isinstance(img, list):
            raise NotImplementedError

        # 이미지가 리스트가 아닌 경우 메서드를 호출해 단일 이미지에 대한 forward pass를 수행하고 결과를 반환
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    # 싱글 이미지에 대해 forward pass를 수행하는 부분
    # 입력 데이터들을 모델의 각 모듈을 통해 forward pass 시키고 예측 결과 및 손실값을 계산해 반환한다.
    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
            self,
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            depths=None,
            radar=None,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        # 입력 데이터를 각 센서에 대해 처리하고, 해당 센서의 특성을 추출하는 부분
        # 각 센서의 종류에 따라 입력 데이터를 처리하고, 각 센서에 맞는 특성을 추출한다.
        # 각각 맞는 메서드를 호출해 특성울 추출하고, 추출된 특성은 features 리스트에 저장된다.
        features = []
        auxiliary_losses = {}
        for sensor in (
                # self.encoders : 각 센서에 대한 인코더 모듈들
                # self.training : 모델이 훈련중인지 여부
                self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                # self.use_depth_loss : depth loss를 사용하는지 여부
                # self.use_depth_loss가 Ture인 경우 카메라 센서 피처 추출 결과에서 depth loss와 피처를 구분해 저장한다.
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        # 만약 모델 훈련 중이 아니라면 features 리스트를 역순으로 정렬한다.
        # 테스트나 추론 시 각 센서의 피처를 정확한 순서로 사용하기 위함이다.
        if not self.training:
            # avoid OOM
            features = features[::-1]

        # 만약 self.fuser가 None이 아닌 경우, features를 결합, 통합하여 최종 피처를 얻는다.
        # 만약 self.fuser가 존재하지 않는 경우, features 리스트의 길이가 1이여야하고, 이 경우에는 첫번째 피처를 사용한다.
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        # 피처 x를 self.decoder 모듈을 통해 디코딩한다.
        # 디코딩 된 피처 x는 backbone, neck 모듈을 거쳐 각각 과정을 수행한다.
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        # 모델 훈련 중인 경우
        if self.training:
            outputs = {}

            # self.heads.items : 다양한 유형의 헤드 모듈들
            # x : 디코딩 된 피처
            # matas : 이미지 메타 데이터
            # gt_bboxes_3d, gt_labels_3d, gt_masks_bev : 훈련용 GT 바운딩 박스, 라벨 2D 지도
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:

                        # self.loss_scale : 손실 스케일링 계수
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val

            # self.use_depth_loss : depth loss 사용 여부
            if self.use_depth_loss:

                # auxiliary_losses : 보조 loss 정보, depth loss
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

