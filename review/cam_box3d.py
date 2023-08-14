import numpy as np
import torch

from mmdet3d.core.points import BasePoints
from .base_box3d import BaseInstance3DBoxes
from .utils import limit_period, rotation_3d_in_axis

class CameraInstance3DBoxes(BaseInstance3DBoxes):
    """
    3D boxes of instances in CAM coordinates.

    Coordinates in camera:

    .. code-block:: none

                z front (yaw=-0.5*pi)
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of z.

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
    카메라 좌표계에서의 좌표:
    
    .. code-block:: none
    
                z 앞 (yaw=-0.5*pi)
               /
              /
             0 ------> x 오른쪽 (yaw=0)
             |
             |
             v
        아래 y
    
    CAM 박스의 바닥 중심의 상대적인 좌표는 (0.5, 1.0, 0.5)이다. 
    그리고 "yaw" 값은 y 축을 중심으로 회전하는 것으로, 회전 축은 y 축(수직 방향)이다. 
    "yaw" 값은 x 축의 양의 방향에서 0이며, x 축의 양의 방향에서 z 축의 양의 방향으로 감소한다. 
    
    현재 세 개의 좌표 체계를 이해하고 서로 변환하기 쉽도록 리팩토링 중이다. 
    
    속성:
        tensor (torch.Tensor): N x box_dim의 부동 소수점 행렬.
        box_dim (int): 박스의 차원을 나타내는 정수
            각 행은 (x, y, z, x 크기, y 크기, z 크기, yaw, ...) 형식
        with_yaw (bool): 참이면 "yaw" 값이 최소/최대 값을 설정
            (minmax 박스 형태)."
    """

    # 생성자
    # tensor : 3D 상자의 정보를 담고 있는 tensor
    # box_dim = 7 : 세 개의 위치 정보(x, y, z), 세 개의 크기 정보(width, height, depth), 하나의 회전 각도(yaw)
    # origin : 상자의 중심을 나타내는 기준점, default = (0.5, 1.0, 0.5)
    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 1.0, 0.5)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 1.0, 0.5):
            dst = self.tensor.new_tensor((0.5, 1.0, 0.5))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    # 상자의 높이 정보를 나타내는 벡터를 반환한다.
    @property
    def height(self):
        """torch.Tensor: A vector with height of each box."""
        return self.tensor[:, 4]

    # 상자의 윗면 높이 정보를 나타내는 벡터를 반환한다.
    @property
    def top_height(self):
        """torch.Tensor: A vector with the top height of each box."""
        # the positive direction is down rather than up
        return self.bottom_height - self.height

    # 상자의 아랫면 높이 정보를 나타내는 벡터를 반환한다.
    @property
    def bottom_height(self):
        """torch.Tensor: A vector with bottom's height of each box."""
        return self.tensor[:, 1]

    # 상자의 중심 위치 정보를 나타내는 텐서를 반환한다.
    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]]
        gravity_center[:, 1] = bottom_center[:, 1] - self.tensor[:, 4] * 0.5
        return gravity_center

    # 상자의 모서리 좌표 정보를 나타내는 텐서를 반환한다.
    # 상자의 모서리는 시계 방향 순서, (N, 8, 3)의 형태의 텐서로 반환된다.
    # 왼쪽 아래 앞 모서리 (x0y0z0)
    # 왼쪽 아래 뒤 모서리 (x0y0z1)
    # 왼쪽 위 뒤 모서리 (x0y1z1)
    # 왼쪽 위 앞 모서리 (x0y1z0)
    # 오른쪽 아래 앞 모서리 (x1y0z0)
    # 오른쪽 아래 뒤 모서리 (x1y0z1)
    # 오른쪽 위 뒤 모서리 (x1y1z1)
    # 오른쪽 위 앞 모서리 (x1y1z0)
    # 이 함수에서 사용되는 rotation_3d_in_axis 함수를 통해 각 모서리의 좌표를 회전하여 박스의 회전을 반영한다.
    # 최종적으로 각 모서리의 좌표는 박스의 위치 정보와 합산되어 모서리의 3D 좌표가 계산
    # 이렇게 계산된 모서리 좌표들을 모아 (N, 8, 3) 형태의 텐서로 반환하게 된다.
    @property
    def corners(self):
        """
        torch.Tensor: Coordinates of corners of all the boxes in
                         shape (N, 8, 3).

        Convert the boxes to  in clockwise order, in the form of
        (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)

        .. code-block:: none

                         front z
                              /
                             /
               (x0, y0, z1) + -----------  + (x1, y0, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                         |  /      .   |  /
                         | / origin    | /
            (x0, y1, z0) + ----------- + -------> x right
                         |             (x1, y1, z0)
                         |
                         v
                    down y
        """

        # TODO: rotation_3d_in_axis function do not support
        # empty tensor currently.
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        ).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 1, 0.5]
        corners_norm = corners_norm - dims.new_tensor([0.5, 1, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # y축 중심으로 회전
        # rotate around y axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=1)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    # 2D BEV 정보를 나타내는 텐서를 XYWHR 형태로 반환한다.
    @property
    def bev(self):
        """torch.Tensor: A n x 5 tensor of 2D BEV box of each box
        with rotation in XYWHR format."""
        return self.tensor[:, [0, 2, 3, 5, 6]]

    # 2D BEV 정보를 반환한다. 이 정보는 회전 없이 바로 사용할 수 있도록 정렬된 형태를 반환한다.
    @property
    def nearest_bev(self):
        """torch.Tensor: A tensor of 2D BEV box of each box
        without rotation."""
        # Obtain BEV boxes with rotation in XZWHR format
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(
            conditions, bev_rotated_boxes[:, [0, 1, 3, 2]], bev_rotated_boxes[:, :4]
        )

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    # 상자를 주어진 각도나 회전 행렬로 회전시키는 기능을 제공
    # angle : 회전 각도, 회전 행렬
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

        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        assert (
            angle.shape == torch.Size([3, 3]) or angle.numel() == 1
        ), f"invalid rotation angle shape {angle.shape}"

        if angle.numel() == 1:
            rot_sin = torch.sin(angle)
            rot_cos = torch.cos(angle)
            rot_mat_T = self.tensor.new_tensor(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]]
            )
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[2, 0]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)

        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        self.tensor[:, 6] += angle

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                # clockwise
                points.rotate(-angle)
            else:
                raise ValueError
            return points, rot_mat_T

    # 상자를 BEV 평면에서 주어진 방향으로 뒤집는 기능을 제공
    # bev_direction : 뒤집을 방향을 나타내며 horizontal or vertical 값이 가능하다.
    def flip(self, bev_direction="horizontal", points=None):
        """
        Flip the boxes in BEV along given BEV direction.

        In CAM coordinates, it flips the x (horizontal) or z (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, None):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """

        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == "vertical":
            self.tensor[:, 2::7] = -self.tensor[:, 2::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == "horizontal":
                    points[:, 0] = -points[:, 0]
                elif bev_direction == "vertical":
                    points[:, 2] = -points[:, 2]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    # 주어진 범위 내 상자가 있는지 확인하는 기능
    # box_range : 상자가 있는지 확인할 범위를 나타낸다.
    def in_range_bev(self, box_range):
        """
        Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                (x_min, z_min, x_max, z_max).

        Note:
            The original implementation of SECOND checks whether boxes in
            a range by checking whether the points are in a convex
            polygon, we reduce the burden for simpler cases.

        Returns:
            torch.Tensor: Indicating whether each box is inside \
                the reference range.
        """

        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 2] > box_range[1])
            & (self.tensor[:, 0] < box_range[2])
            & (self.tensor[:, 2] < box_range[3])
        )
        return in_range_flags

    # height_overlaps 클래스 메서드는 두 상자의 높이 영역 겹침을 계산
    # boxes1과 boxes2는 각 높이 영역을 갖는 상자로, 같은 타입이여야 한다.
    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode="iou"):
        """
        Calculate height overlaps of two boxes.

        This function calculates the height overlaps between ``boxes1`` and
        ``boxes2``, where ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`CameraInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`CameraInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes' heights.
        """

        assert isinstance(boxes1, CameraInstance3DBoxes)
        assert isinstance(boxes2, CameraInstance3DBoxes)

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        # In camera coordinate system
        # from up to down is the positive direction
        heighest_of_bottom = torch.min(boxes1_bottom_height, boxes2_bottom_height)
        lowest_of_top = torch.max(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(heighest_of_bottom - lowest_of_top, min=0)
        return overlaps_h

    # 다른 상자 모드로 변환하는 기능을 제공
    # dst : 변환할 상자 모드
    # rt_mat : 변환에 필요한 회전 및 이동 행렬
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
            :obj:`BaseInstance3DBoxes`:  \
                The converted box of the same type in the ``dst`` mode.
        """

        from .box_3d_mode import Box3DMode

        return Box3DMode.convert(box=self, src=Box3DMode.CAM, dst=dst, rt_mat=rt_mat)
