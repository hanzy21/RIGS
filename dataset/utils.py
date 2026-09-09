import numpy as np
from pyquaternion import Quaternion
import torch
import numbers


def get_rm(angle, axis, deg=False):
    if deg:
        angle = np.deg2rad(angle)
    rm = np.eye(3)
    if axis == 'x':
        rm[1, 1] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[1, 2] = - np.sin(angle)
        rm[2, 1] = np.sin(angle)
    elif axis == 'y':
        rm[0, 0] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[0, 2] = np.sin(angle)
        rm[2, 0] = - np.sin(angle)
    elif axis == 'z':
        rm[0, 0] = np.cos(angle)
        rm[1, 1] = np.cos(angle)
        rm[0, 1] = - np.sin(angle)
        rm[1, 0] = np.sin(angle)
    return rm


def get_xyz(pose_dict):
    return np.array(pose_dict['translation'])

def get_img2global(calib_dict, pose_dict):
    
    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(calib_dict['camera_intrinsic'])
    # 注意：这里img2cam的计算可能不正确，因为K是投影矩阵
    # 正确的做法应该是直接构建lidar2img，而不是通过img2global
    img2cam = np.linalg.inv(cam2img)

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    img2global = ego2global @ cam2ego @ img2cam
    return img2global

def get_lidar2global(calib_dict, pose_dict):

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    lidar2global = ego2global @ lidar2ego
    return lidar2global


def custom_collate_fn_temporal(instances):
    return_dict = {}
    for k, v in instances[0].items():
        # Unix epoch timestamps are around 1e9 seconds. Converting them to
        # PyTorch's default float32 destroys sub-second frame intervals (the
        # float32 spacing is 128 seconds at K-Radar timestamps), which makes
        # VelocityLoss see delta_t == 0 and skip every temporal target.
        if k in {"timestamp", "prev_timestamp"}:
            values = [instance[k] for instance in instances]
            return_dict[k] = (
                values if any(value is None for value in values)
                else torch.as_tensor(values, dtype=torch.float64)
            )
            continue
        if isinstance(v, np.ndarray):
            all_arrays = all(isinstance(inst[k], np.ndarray) for inst in instances)
            if all_arrays:
                return_dict[k] = torch.stack([
                    torch.from_numpy(instance[k].copy()) for instance in instances])
            else:
                return_dict[k] = [
                    torch.from_numpy(inst[k].copy()) if isinstance(inst[k], np.ndarray)
                    else None
                    for inst in instances
                ]
        elif isinstance(v, torch.Tensor):
            return_dict[k] = torch.stack([instance[k] for instance in instances])
        elif isinstance(v, (dict, str)):
            return_dict[k] = [instance[k] for instance in instances]
        elif v is None:
            return_dict[k] = [None] * len(instances)
        elif isinstance(v, bool):
            # 处理布尔值（如prev_frame_available, is_continuous_frame等）
            return_dict[k] = [instance[k] for instance in instances]
        elif isinstance(v, numbers.Real):
            return_dict[k] = torch.as_tensor([instance[k] for instance in instances])
        else:
            raise NotImplementedError(f"Unsupported type for key '{k}': {type(v)}")
    return return_dict
