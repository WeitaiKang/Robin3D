import numpy as np
import torch

def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def roty_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(
        tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device
    )
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output

def get_3d_box_batch_lwh_tensor(box_size, angle, center):
    assert isinstance(box_size, torch.Tensor)
    assert isinstance(angle, torch.Tensor)
    assert isinstance(center, torch.Tensor)

    reshape_final = False
    if angle.ndim == 2:
        assert box_size.ndim == 3
        assert center.ndim == 3
        bsize = box_size.shape[0]
        nprop = box_size.shape[1]
        box_size = box_size.reshape(-1, box_size.shape[-1])
        angle = angle.reshape(-1)
        center = center.reshape(-1, 3)
        reshape_final = True

    input_shape = angle.shape
    R = roty_batch_tensor(angle)
    l = torch.unsqueeze(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = torch.unsqueeze(box_size[..., 1], -1)
    h = torch.unsqueeze(box_size[..., 2], -1)
    corners_3d = torch.zeros(
        tuple(list(input_shape) + [8, 3]), device=box_size.device, dtype=torch.float32
    )
    corners_3d[..., :, 0] = torch.cat(
        (l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1
    )
    corners_3d[..., :, 1] = torch.cat(
        (w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1
    )
    corners_3d[..., :, 2] = torch.cat(
        (h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1
    )
    
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = torch.matmul(corners_3d, R.permute(tlist))
    corners_3d += torch.unsqueeze(center, -2)
    if reshape_final:
        corners_3d = corners_3d.reshape(bsize, nprop, 8, 3)
    return corners_3d

def get_box3d_min_max_batch_tensor(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: PyTorch tensor (N,8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an tensor for min and max coordinates of 3D bounding box IoU

    '''

    min_coord, _ = corner.min(dim=1)
    max_coord, _ = corner.max(dim=1)
    x_min, x_max = min_coord[:, 0], max_coord[:, 0]
    y_min, y_max = min_coord[:, 1], max_coord[:, 1]
    z_min, z_max = min_coord[:, 2], max_coord[:, 2]
    
    return x_min, x_max, y_min, y_max, z_min, z_max

def box3d_iou_batch_tensor(corners1, corners2):
    ''' Compute 3D bounding box IoU.
        Note: only for axis-aligned bounding boxes

    Input:
        corners1: PyTorch tensor (N,8,3), assume up direction is Z (batch of N samples)
        corners2: PyTorch tensor (N,8,3), assume up direction is Z (batch of N samples)
    Output:
        iou: an tensor of 3D bounding box IoU (N)

    '''
    
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max_batch_tensor(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max_batch_tensor(corners2)
    xA = torch.max(x_min_1, x_min_2)
    yA = torch.max(y_min_1, y_min_2)
    zA = torch.max(z_min_1, z_min_2)
    xB = torch.min(x_max_1, x_max_2)
    yB = torch.min(y_max_1, y_max_2)
    zB = torch.min(z_max_1, z_max_2)
    # zeros = corners1.new_zeros(xA.shape).to(xA.device)
    zeros = torch.zeros(xA.shape, dtype=corners1.dtype, device=corners1.device)
    inter_vol = torch.max((xB - xA), zeros) * torch.max((yB - yA), zeros) * torch.max((zB - zA), zeros)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou