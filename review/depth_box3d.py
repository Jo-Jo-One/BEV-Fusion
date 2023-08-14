import numpy as np
import torch

from mmdet3d.core.points import BasePoints
from mmdet3d.ops import points_in_boxes_batch
from .base_box3d import BaseInstance3DBoxes
from .utils import limit_period, rotation_3d_in_axis

class DepthInstance3DBoxes(BaseInstance3DBoxes):
    """
    3D boxes of instances in Depth coordinates.

    Coordinates in Depth:

    .. code-block:: none

                    up z    y front (yaw=-0.5*pi)
                       ^   ^
                       |  /
                       | /
                       0 ------> x right (yaw=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of y.
    Also note that rotation of DepthInstance3DBoxes is counterclockwise,
    which is reverse to the definition of the yaw angle (clockwise).

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    """
     깊이 좌표계에서의 좌표:

     .. code-block:: none

                     위로 z    y 앞쪽 (yaw=-0.5*pi)
                        ^   ^
                        |  /
                        | /
                        0 ------> x 오른쪽 (yaw=0)

     깊이 상자의 하단 중심의 상대 좌표는 (0.5, 0.5, 0)이며,
     yaw 값은 z 축을 중심으로 회전하며, 회전 축은 2이다.
     yaw 값은 x 축의 양방향에서 0이며, x의 양방향에서 y의 양방향으로 감소한다.
     또한 깊이 좌표계에서의 회전은 시계 반대방향이며,
     이는 yaw 각도의 정의(시계 방향)와 반대이다. 

     세 좌표계 간의 변환 및 이해를 더 쉽게하기 위한 리팩토링이 진행 중이다.

     속성:
         tensor (torch.Tensor): N x box_dim의 부동 소수점 행렬
         box_dim (int): 상자의 차원을 나타내는 정수
             각 행은 (x, y, z, x 크기, y 크기, z 크기, yaw, ...)이다. 
         with_yaw (bool): True인 경우, yaw 값을 최소/최대 상자로 설정
     """

    # 각 상자의 중심 좌표를 계산하여 반환, 하단 중심 좌표와 높이 정보를 이용한다.
    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    # 상자의 모서리 좌표 정보를 나타내는 텐서를 반환한다.
    # 상자의 모서리는 시계 방향 순서, (N, 8, 3)의 형태의 텐서로 반환된다.
    # 모서리 좌표는 상자의 차원 정보 및 회전 정보를 이용하여 계산된다.
    @property
    def corners(self):
        """
        torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front y           ^
                                 /            |
                                /             |
                  (x0, y1, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- + --------> right x
                                          (x1, y0, z0)
        """

        # TODO: rotation_3d_in_axis function do not support
        #  empty tensor currently.
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        ).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0.5, 0)
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # z축 중심으로 회전
        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=2)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    # 각 상자의 2D BEV 상자 정보를 나타내는 n X 5 텐서, XYWHR 형식
    @property
    def bev(self):
        """torch.Tensor: A n x 5 tensor of 2D BEV box of each box
        in XYWHR format."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    # 각 상자의 2D BEV 상자 정보를 (n, 5) 텐서 형태로 반환한다.
    # nearest_bev는 회전 없는 2D BEV 상자 정보를 계산하여 반환한다.
    # 상자 회전을 고려해 중심값을 찾고, 필요에 따하 회전 각도를 조정해 BEV 상자 정보를 계산한다.
    @property
    def nearest_bev(self):
        """torch.Tensor: A tensor of 2D BEV box of each box
        without rotation."""
        # 회전 포함 XYWHR 형식의 BEV 상자 가져오기
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # 상자의 중심 찾기
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(
            conditions, bev_rotated_boxes[:, [0, 1, 3, 2]], bev_rotated_boxes[:, :4]
        )

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    # 상자와 점을 주어진 각도 또는 회전 행렬로 회전시키는 기능을 제공
    # 회전하는 경우, 상자의 줌심 및 모서리 좌표를 조정하고 회전 각도를 반영한다.
    # 회전한 경우, 점들도 회전하고 회전 행렬을 반환한다.
    def rotate(self, angle, points=None):
        """
        Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns \
                None, otherwise it returns the rotated points and the \
                rotation matrix ``rot_mat_T``.
        """

        """
        각도 또는 회전 행렬로 상자와 점(선택 사항)을 회전

                Args:
                    angle (float | torch.Tensor | np.ndarray):
                        회전 각도 또는 회전 행렬.
                    points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                        회전할 점들. 기본값은 None

                Returns:
                    tuple 또는 None: 만약 points가 None이면, 함수는 None을 반환하며,
                        그렇지 않으면 회전된 점과 회전 행렬 rot_mat_T를 반환
        """

        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        assert (
            angle.shape == torch.Size([3, 3]) or angle.numel() == 1
        ), f"invalid rotation angle shape {angle.shape}"

        if angle.numel() == 1:
            rot_sin = torch.sin(angle)
            rot_cos = torch.cos(angle)
            rot_mat_T = self.tensor.new_tensor(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
            ).T
        else:
            rot_mat_T = angle.T
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)

        self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T
        if self.with_yaw:
            self.tensor[:, 6] -= angle
        else:
            corners_rot = self.corners @ rot_mat_T
            new_x_size = (
                corners_rot[..., 0].max(dim=1, keepdim=True)[0]
                - corners_rot[..., 0].min(dim=1, keepdim=True)[0]
            )
            new_y_size = (
                corners_rot[..., 1].max(dim=1, keepdim=True)[0]
                - corners_rot[..., 1].min(dim=1, keepdim=True)[0]
            )
            self.tensor[:, 3:5] = torch.cat((new_x_size, new_y_size), dim=-1)

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                # 반시계 방향 회전
                points.rotate(angle)
            else:
                raise ValueError
            return points, rot_mat_T

    # 주어진 BEV 방향을 기준으로 상자와 점을 뒤집는 기능을 제공
    # 수평 방향으로 뒤집는 경우 x 좌표가 반전되고, 수직 방향으로 뒤집는 경우 y 좌표가 반전된다.
    # 필요한 경우 점들도 뒤집힌 좌표를 반환한다.
    def flip(self, bev_direction="horizontal", points=None):
        """
        Flip the boxes in BEV along given BEV direction.

        In Depth coordinates, it flips x (horizontal) or y (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, None):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """

        """
        주어진 BEV 방향을 기준으로 상자를 뒤집는다. 
        
        깊이 좌표계에서는 x(수평) 또는 y(수직) 축을 뒤집는다. 
        
        Args:
            bev_direction (str): 뒤집을 방향 (horizontal 또는 vertical).
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, None):
                뒤집을 점들. 기본값은 None
        
        Returns:
            torch.Tensor, numpy.ndarray 또는 None: 뒤집힌 점들
        """

        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == "vertical":
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == "horizontal":
                    points[:, 0] = -points[:, 0]
                elif bev_direction == "vertical":
                    points[:, 1] = -points[:, 1]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    # 각 상자가 주어진 범위 내에 있는지 확인하는 기능을 제공한다.
    # 주어진 상자 범위의 최소 및 최대 좌표를 비교해 상자가 범위 내 있는지 여부를 계산한다.
    def in_range_bev(self, box_range):
        """
        Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                (x_min, y_min, x_max, y_max).

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burdun for simpler cases.

        Returns:
            torch.Tensor: Indicating whether each box is inside \
                the reference range.
        """

        """
        상자가 주어진 범위 내에 있는지 확인
        
        Args:
            box_range (list | torch.Tensor): 상자의 범위 (x_min, y_min, x_max, y_max).
        
        Note:
            SECOND의 원래 구현에서, 상자가 범위 내에 있는지 확인하기 위해
            점이 볼록 다각형 내에 있는지 확인
        
        Returns:
            torch.Tensor: 각 상자가 참조 범위 내에 있는지를 나타내는 인덱스
        """

        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 0] < box_range[2])
            & (self.tensor[:, 1] < box_range[3])
        )
        return in_range_flags

    # 현재 상자를 다른 모드로 변환하는 기능을 제공
    # 변환할 대상 모드와 필요한 경우 변환 행렬을 지정해 변환을 수행한다.
    # 변환 후, 지정된 대상 모드에서 상자를 반환한다.
    def convert_to(self, dst, rt_mat=None):
        """
        Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`DepthInstance3DBoxes`: \
                The converted box of the same type in the ``dst`` mode.
        """

        """
        self를 ``dst`` 모드로 변환

        Args:
            dst (:obj:`Box3DMode`): 대상 Box 모드
            rt_mat (np.ndarray | torch.Tensor): 서로 다른 좌표간의 회전 및 변환 행렬
                기본값은 None
                보통, 다른 센서로의 변환 (예: 카메라에서 LiDAR로)은 변환 행렬이 필요

        Returns:
            :obj:`DepthInstance3DBoxes`: ``dst`` 모드에서 동일한 타입의 변환된 상자
        """

        from .box_3d_mode import Box3DMode

        return Box3DMode.convert(box=self, src=Box3DMode.DEPTH, dst=dst, rt_mat=rt_mat)

    # 주어진 상자 내에 있는 점들을 찾는 기능을 제공
    # 입력된 점들을 LiDAR 좌표계로 변환한 뒤, 변환된 LiDAR 좌표계에서의 상자와 점들 간의 관계를 계산해 점이 어느 상자에 속하는지 인덱스로 반환한다.
    def points_in_boxes(self, points):
        """
        Find points that are in boxes (CUDA).

        Args:
            points (torch.Tensor): Points in shape [1, M, 3] or [M, 3], \
                3 dimensions are [x, y, z] in LiDAR coordinate.

        Returns:
            torch.Tensor: The index of boxes each point lies in with shape \
                of (B, M, T).
        """

        """
        상자 내에 있는 점들을 찾는다. (CUDA 사용).

        Args:
            points (torch.Tensor): 점들 (LiDAR 좌표계에서 [1, M, 3] 또는 [M, 3]).

        Returns:
            torch.Tensor: 각 점이 속한 상자의 인덱스를 나타내는 텐서 (B, M, T).
        """

        from .box_3d_mode import Box3DMode

        # to lidar
        points_lidar = points.clone()
        points_lidar = points_lidar[..., [1, 0, 2]]
        points_lidar[..., 1] *= -1
        if points.dim() == 2:
            points_lidar = points_lidar.unsqueeze(0)
        else:
            assert points.dim() == 3 and points_lidar.shape[0] == 1

        boxes_lidar = self.convert_to(Box3DMode.LIDAR).tensor
        boxes_lidar = boxes_lidar.to(points.device).unsqueeze(0)
        box_idxs_of_pts = points_in_boxes_batch(points_lidar, boxes_lidar)

        return box_idxs_of_pts.squeeze(0)

    # 상자의 길이, 너비 및 높이를 확장하는 기능을 제공
    # 상자의 크기를 지정된 추가 너비의 2배만큼 확장시키고, 밑면의 중심 좌표인 z 좌표를 extra_width만큼 감소시킨다.
    # 확장된 상자를 반환한다.
    def enlarged_box(self, extra_width):
        """
        상자의 길이, 너비 및 높이를 확장

        Args:
            extra_width (float | torch.Tensor): 상자를 확장하기 위한 추가 너비

        Returns:
            :obj:`LiDARInstance3DBoxes`: 확장된 상자
        """

        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # 밑면 중심의 z를 extra_width만큼 감소시킨다.
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    # 상자의 표면과 선의 중심을 계산하는 기능을 제공한다.
    # 상자의 크기 및 중심 좌표, 그리고 회전된 상자의 회전 각도와 관련된 회전 행렬을 사용해 표면 중심과 선의 중심 좌표를 계산
    # 표면 중심은 각 변의 중심을 의미, 선의 중심은 상자의 각 모서리에서 대각선을 그은 중심을 의미
    def get_surface_line_center(self):
        """
        상자의 표면과 선의 중심을 계산한다.

        Returns:
            torch.Tensor: 상자의 표면과 선의 중심
        """

        obj_size = self.dims
        center = self.gravity_center.view(-1, 1, 3)
        batch_size = center.shape[0]

        rot_sin = torch.sin(-self.yaw)
        rot_cos = torch.cos(-self.yaw)
        rot_mat_T = self.yaw.new_zeros(tuple(list(self.yaw.shape) + [3, 3]))
        rot_mat_T[..., 0, 0] = rot_cos
        rot_mat_T[..., 0, 1] = -rot_sin
        rot_mat_T[..., 1, 0] = rot_sin
        rot_mat_T[..., 1, 1] = rot_cos
        rot_mat_T[..., 2, 2] = 1

        # Get the object surface center
        offset = obj_size.new_tensor(
            [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
        )
        offset = offset.view(1, 6, 3) / 2
        surface_3d = (offset * obj_size.view(batch_size, 1, 3).repeat(1, 6, 1)).reshape(-1, 3)

        # Get the object line center
        offset = obj_size.new_tensor(
            [
                [1, 0, 1],
                [-1, 0, 1],
                [0, 1, 1],
                [0, -1, 1],
                [1, 0, -1],
                [-1, 0, -1],
                [0, 1, -1],
                [0, -1, -1],
                [1, 1, 0],xf
                [1, -1, 0],
                [-1, 1, 0],
                [-1, -1, 0],
            ]
        )
        offset = offset.view(1, 12, 3) / 2

        line_3d = (offset * obj_size.view(batch_size, 1, 3).repeat(1, 12, 1)).reshape(-1, 3)

        surface_rot = rot_mat_T.repeat(6, 1, 1)
        surface_3d = torch.matmul(surface_3d.unsqueeze(-2), surface_rot.transpose(2, 1)).squeeze(-2)
        surface_center = center.repeat(1, 6, 1).reshape(-1, 3) + surface_3d

        line_rot = rot_mat_T.repeat(12, 1, 1)
        line_3d = torch.matmul(line_3d.unsqueeze(-2), line_rot.transpose(2, 1)).squeeze(-2)
        line_center = center.repeat(1, 12, 1).reshape(-1, 3) + line_3d

        return surface_center, line_center
