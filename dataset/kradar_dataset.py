import os
from copy import deepcopy
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

import mmengine
from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class KRadarDataset(Dataset):

    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        num_samples=0,
        vis_scene_index=-1,
        phase='train',
        return_keys=[
            'img',
            'projection_mat',
            'image_wh',
            'occ_label',
            'occ_xyz',
            'occ_cam_mask',
            'ori_img',
            'cam_positions',
            'focal_positions'
        ],
    ):
        self.data_path = data_root
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']
        # 保持原始顺序（按场景连续排列），不按 frame_index 排序
        # sorted(key=lambda x: x[1]) 会把不同场景的帧交错排列，
        # 破坏时序连续性，导致 temporal context 几乎完全失效
        # self.keyframes = sorted(self.keyframes, key=lambda x: x[1])

        self.data_aug_conf = data_aug_conf
        self.test_mode = (phase != 'train')

        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

        # KRadar has 4 cameras: front, rear, left, right
        self.sensor_types = ['CAM_FRONT']#['CAM_FRONT', 'CAM_REAR', 'CAM_LEFT', 'CAM_RIGHT']
        self.return_keys = return_keys
        if vis_scene_index >= 0:
            frame = self.keyframes[vis_scene_index]
            num_frames = len(self.scene_infos[frame[0]])
            self.keyframes = [(frame[0], i) for i in range(num_frames)]
            print(f'Scene length: {len(self.keyframes)}')
        elif vis_indices is not None:
            if len(vis_indices) > 0:
                vis_indices = [i % len(self.keyframes) for i in vis_indices]
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
            elif num_samples > 0:
                vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
        elif num_samples > 0:
            vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
            self.keyframes = [self.keyframes[idx] for idx in vis_indices]

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __getitem__(self, index):
        # index is the global index into self.keyframes
        # keyframes[index] returns (scene_token, frame_index) where frame_index is local to that scene
        scene_token, frame_index = self.keyframes[index]
        frame_index = frame_index - 1
        
        if scene_token not in self.scene_infos:
            raise KeyError(f"scene {scene_token!r} is missing from the index")
        if not 0 <= frame_index < len(self.scene_infos[scene_token]):
            raise IndexError(f"frame {frame_index + 1} is outside scene {scene_token!r}")
        info = deepcopy(self.scene_infos[scene_token][frame_index])
        
        # ========== 时序初始化：从 scene_infos 直接加载时间上一帧 ==========
        # 不依赖数据集采样顺序：直接访问 scene_infos[scene_token][frame_index - 1]
        prev_frame_available = False
        is_same_scene = True  # 单场景内必然 same scene
        is_continuous_frame = False
        prev_ego2global = None
        prev_info = None
        
        if 'previous_token' not in info:
            raise KeyError(
                "runtime index lacks previous_token; regenerate it from the canonical JSONL manifest"
            )
        expected_previous_token = info['previous_token']
        if expected_previous_token is not None:
            if frame_index == 0:
                raise ValueError(
                    f"frame {info.get('token')} declares a predecessor at the start of a scene"
                )
            candidate = self.scene_infos[scene_token][frame_index - 1]
            if candidate.get('token') != expected_previous_token:
                raise ValueError(
                    f"non-adjacent predecessor for {info.get('token')}: "
                    f"expected {expected_previous_token}, found {candidate.get('token')}"
                )
            prev_info = deepcopy(candidate)
            prev_ego2global = get_lidar2global(
                prev_info['data']['LIDAR_TOP']['calib'],
                prev_info['data']['LIDAR_TOP']['pose']
            )
            prev_frame_available = True
            is_continuous_frame = True
        
        input_dict = self.get_data_info(info)
        
        # 添加当前帧的ego2global
        curr_ego2global = get_lidar2global(
            info['data']['LIDAR_TOP']['calib'],
            info['data']['LIDAR_TOP']['pose']
        )
        input_dict['curr_ego2global'] = curr_ego2global.astype(np.float32)
        
        # 添加上一帧的位姿信息（如果可用）
        if prev_frame_available and prev_ego2global is not None:
            input_dict['prev_ego2global'] = prev_ego2global.astype(np.float32)
            input_dict['prev_frame_available'] = True
            input_dict['is_same_scene'] = True
            input_dict['is_continuous_frame'] = True
            # G1: 提取上一帧的 OCC 标签路径（用于时序位移解混叠）
            prev_occ_path = prev_info.get("occ_path", "") if prev_info else ""
            if prev_occ_path and not os.path.isabs(prev_occ_path):
                prev_occ_path = os.path.join(self.data_path, prev_occ_path)
            input_dict['prev_occ_path'] = prev_occ_path if prev_occ_path else None
            input_dict['prev_timestamp'] = prev_info.get('timestamp')
        else:
            input_dict['prev_ego2global'] = None
            input_dict['prev_frame_available'] = False
            input_dict['is_same_scene'] = is_same_scene
            input_dict['is_continuous_frame'] = False
            input_dict['prev_occ_path'] = None
            input_dict['prev_timestamp'] = None

        #TODO: 数据增强先去掉了
        # if self.data_aug_conf is not None:
        #     input_dict["aug_configs"] = self._sample_augmentation()
        
        # Keep Doppler aliased here. Eq. (12) resolves folding in VelocityLoss.

        for t in self.pipeline:
            input_dict = t(input_dict)
        
        return_dict = {k: input_dict[k] for k in self.return_keys}
        # 确保位姿信息也被返回（即使不在return_keys中）
        if 'prev_ego2global' in input_dict:
            return_dict['prev_ego2global'] = input_dict['prev_ego2global']
        if 'curr_ego2global' in input_dict:
            return_dict['curr_ego2global'] = input_dict['curr_ego2global']
        if 'prev_frame_available' in input_dict:
            return_dict['prev_frame_available'] = input_dict['prev_frame_available']
        if 'is_same_scene' in input_dict:
            return_dict['is_same_scene'] = input_dict['is_same_scene']
        if 'is_continuous_frame' in input_dict:
            return_dict['is_continuous_frame'] = input_dict['is_continuous_frame']
        # 确保radar_points也被返回（即使不在return_keys中）
        if 'radar_points' in input_dict:
            return_dict['radar_points'] = input_dict['radar_points']
        if 'radar_intensity' in input_dict:
            return_dict['radar_intensity'] = input_dict['radar_intensity']
        if 'radar_velocity' in input_dict:
            return_dict['radar_velocity'] = input_dict['radar_velocity']
        if 'radar_features' in input_dict:
            return_dict['radar_features'] = input_dict['radar_features']
        # G1: 返回上一帧 OCC 标签（用于时序位移解混叠）
        if 'prev_occ_label' in input_dict:
            return_dict['prev_occ_label'] = input_dict['prev_occ_label']
        return_dict['timestamp'] = input_dict.get('timestamp')
        return_dict['prev_timestamp'] = input_dict.get('prev_timestamp')
        return_dict['sample_idx'] = input_dict.get('sample_idx')
        return_dict['previous_sample_idx'] = expected_previous_token
        return return_dict
    
    def get_data_info(self, info):
        # f 是相机坐标系中沿光轴方向的距离（米单位）
        # 如果 fx=567.7, fy=577.2（像素单位），物理焦距通常约为 2-8mm
        # 这里使用 fx 和 fy 的平均值，假设图像宽度为 2560，归一化后估算
        # 实际应该使用相机的物理焦距值（从标定文件获取）
        # 如果不知道物理焦距，可以使用默认值 0.0055（约 5.5mm） TODO
        f = 0.00229#0.0055  # 相机焦距位置距离（米），可根据实际相机参数调整
        image_paths = []
        lidar2img_rts = []
        ego2image_rts = []
        cam_positions = []
        focal_positions = []

        # Get lidar calibration and pose
        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        #TODO: 确认这里的calib是否准确
        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['data']['LIDAR_TOP']['pose']['rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['data']['LIDAR_TOP']['pose']['translation']).T

        for cam_type in self.sensor_types:
            # 文件路径可能包含片段名称，直接拼接
            img_path = info['data'][cam_type]['filename']
            resolved_image = img_path if os.path.isabs(img_path) else os.path.join(self.data_path, img_path)
            if not os.path.isfile(resolved_image):
                raise FileNotFoundError(f"missing image for {cam_type}: {resolved_image}")
            image_paths.append(resolved_image)

            # 修正：直接从标定文件（cam_1.yml等）加载lidar2cam参数，构建正确的lidar2img
            calib_dict = info['data'][cam_type]['calib']
            required_calib = {'camera_intrinsic', 'rotation', 'translation'}
            if not required_calib.issubset(calib_dict):
                raise KeyError(f"{cam_type} calibration lacks {required_calib - set(calib_dict)}")
            
            # The canonical index contains sequence-specific camera calibration
            # and pose. External files and guessed sequence defaults are forbidden.
            img2global = get_img2global(calib_dict, info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global

            lidar2img_rts.append(lidar2img)
            
            # 构建img2global用于ego2image_rts
            # img2global = lidar2global @ inv(lidar2img_4x4)
            # 但是lidar2img_4x4的前3行是投影矩阵，不能直接求逆
            # 我们需要通过lidar2cam和cam2ego来构建img2global
            # img2global = ego2global @ cam2ego @ img2cam
            # 其中img2cam需要从K构建，但K是3x3的投影矩阵
            # 实际上，我们可以通过lidar2img和lidar2global的关系来构建
            # img2global = lidar2global @ inv(lidar2img)，但lidar2img是4x4的投影矩阵
            # 更简单的方法：img2global = lidar2global @ cam2lidar @ K^(-1)（扩展到4x4）
            # 但这样还是有问题
            
            # 暂时使用原来的方法构建img2global（虽然有问题，但至少能运行）
            img2global = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)

            img2lidar = np.linalg.inv(lidar2global) @ img2global
            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            cam_position = img2lidar @ viewpad @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = img2lidar @ viewpad @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
            
        # 处理occ_path：如果是相对路径（包含segment名称），转换为绝对路径
        occ_path = info.get("occ_path", "")
        if occ_path and not os.path.isabs(occ_path):
            # 如果occ_path包含segment名称（例如 "10/semantic_occupancy_gt/occupancy_gt_with_semantic1.npy"）
            # 从data_root拼接完整路径
            occ_path_full = os.path.join(self.data_path, occ_path)
        else:
            occ_path_full = occ_path
        
        # ========== 加载 Tesseract 雷达数据 ==========
        radar_points = None
        if 'TESSERACT' in info['data']:
            tesseract_info = info['data']['TESSERACT']
            tesseract_path = tesseract_info['filename']
            
            # 构建完整路径
            if not os.path.isabs(tesseract_path):
                tesseract_full_path = os.path.join(self.data_path, tesseract_path)
            else:
                tesseract_full_path = tesseract_path
            
            # 加载 EAsparse 文件
            if os.path.exists(tesseract_full_path):
                try:
                    tesseract_data = np.load(tesseract_full_path, allow_pickle=False)
                    
                    # EAsparse 文件格式（与 generate_4d_polar_doppler_self.py 一致）：
                    #   range_ind, elevation_ind, azimuth_ind, power_val
                    # power_val 形状 [8, N]，8 维含义为：
                    #   [0:3]  top-3 功率值（log10，来自 arrDREA 在该点多普勒维上的最强 3 个 bin）
                    #   [3:6]  top-3 多普勒 bin 索引（0 ~ d_dim-1，如 0~63 对应 64 个 Doppler bin）
                    #   [6]    该点在所有多普勒 bin 上的功率均值
                    #   [7]    该点在所有多普勒 bin 上的功率方差
                    # 将 (range, elevation, azimuth) 索引转换为 3D 坐标，并解析强度/速度
                    if all(key in tesseract_data for key in ['range_ind', 'elevation_ind', 'azimuth_ind']):
                        range_ind = tesseract_data['range_ind']
                        elevation_ind = tesseract_data['elevation_ind']
                        azimuth_ind = tesseract_data['azimuth_ind']
                        
                        # 读取功率值（反射强度）和速度信息
                        power_val = None
                        if 'power_val' in tesseract_data:
                            power_val = tesseract_data['power_val']  # [8, N]
                            # power_val格式（根据EAsparse生成代码）：
                            # 前3行（索引0-2）：top-3多普勒通道的功率值
                            # 第4-6行（索引3-5）：top-3多普勒通道的索引（0-7）
                            # 第7行（索引6）：均值
                            # 第8行（索引7）：标准差
                        
                        # K-Radar雷达参数
                        range_resolution = 0.46  # 米
                        elevation_size, elevation_resolution = 37, 1.0  # -18° 到 +18°
                        azimuth_size, azimuth_resolution = 107, 1.0  # -53° 到 +53°
                        
                        # K-Radar多普勒速度参数（64个bin）
                        # 多普勒分辨率：0.06 m/s
                        # 总共有64个bin，速度范围通常是对称的：-1.92到+1.92 m/s
                        num_doppler_bins = 64
                        doppler_resolution = 0.06  # m/s
                        max_doppler_velocity = num_doppler_bins * doppler_resolution / 2  # 3.84 / 2 = 1.92 m/s
                        # 速度范围：从 -1.92 到 +1.92 m/s（对称分布）
                        doppler_velocity_centers = np.linspace(
                            -max_doppler_velocity + doppler_resolution / 2,
                            max_doppler_velocity - doppler_resolution / 2,
                            num_doppler_bins
                        )  # [64] 每个bin的中心速度值
                        
                        # 计算角度并转换为弧度
                        elevation_angles = np.linspace(-(elevation_size-1)/2 * elevation_resolution, 
                                                      (elevation_size-1)/2 * elevation_resolution, 
                                                      elevation_size)
                        azimuth_angles = np.linspace(-(azimuth_size-1)/2 * azimuth_resolution, 
                                                     (azimuth_size-1)/2 * azimuth_resolution, 
                                                     azimuth_size)
                        azimuth_rad = np.deg2rad(azimuth_angles[azimuth_ind])
                        elevation_rad = np.deg2rad(elevation_angles[elevation_ind])
                        
                        # 转换为XYZ坐标（雷达坐标系：X-前，Y-左，Z-上）
                        # K-Radar 方位角约定：正方位角 = 车辆右侧（顺时针），
                        # 因此 Y 坐标需要取负号：y = -r * cos(elev) * sin(azimuth)
                        # 参考 K-Radar 官方代码 util_geometry.py:
                        #   get_rdr_pc_from_tesseract:  val_y = val_r*np.cos(val_e)*np.sin(-val_a)
                        #   get_xy_from_ra_color:       azi = np.arctan2(-y, x)
                        ranges = range_ind * range_resolution
                        radar_points_tesseract = np.column_stack((
                            ranges * np.cos(elevation_rad) * np.cos(azimuth_rad),
                            -ranges * np.cos(elevation_rad) * np.sin(azimuth_rad),
                            ranges * np.sin(elevation_rad)
                        ))
                        
                        #TODO： 提取方法需要优化
                        # 提取反射强度和速度信息
                        radar_intensity = None
                        radar_velocity = None
                        if power_val is not None:
                            if power_val.ndim != 2 or power_val.shape[0] != 8:
                                raise ValueError(
                                    f"power_val must be [8,N], got {power_val.shape}")
                            # power_val格式：[8, N]
                            # 索引0-2：top-3功率值
                            # 索引3-5：top-3通道索引（0-63，对应64个多普勒bin）
                            # 索引6：64个多普勒通道的功率均值
                            # 索引7：64个多普勒通道的功率方差
                            
                            # 提取反射强度：使用均值（索引6）或 top-1 功率值（索引0）
                            # 方法1：使用均值（更稳定）
                            radar_intensity = power_val[6, :]  # [N] 均值
                            # 方法2：使用top-1功率值（可选，更敏感）
                            # radar_intensity = power_val[0, :]  # [N] top-1功率值
                            
                            # 提取径向速度：使用top-1多普勒通道索引（索引3）
                            # 这个索引对应功率最强的多普勒通道（0-63，对应64个多普勒bin）
                            top1_doppler_idx = power_val[3, :].astype(np.int32)
                            if ((top1_doppler_idx < 0) | (top1_doppler_idx >= num_doppler_bins)).any():
                                raise ValueError("top-1 Doppler indices must be in [0, 63]")
                            # 获取对应的径向速度
                            radar_velocity = doppler_velocity_centers[top1_doppler_idx]  # [N]
                        else:
                            print(f"Warning: Failed to load radar data from {tesseract_full_path}: power_val is None")
                        
                        # 坐标系转换：Tesseract → Ego → LiDAR
                        tesseract2ego = np.eye(4)
                        tesseract2ego[:3, :3] = Quaternion(tesseract_info['calib']['rotation']).rotation_matrix
                        tesseract2ego[:3, 3] = tesseract_info['calib']['translation']
                        tesseract2lidar = ego2lidar @ tesseract2ego
                        
                        radar_points_homo = np.column_stack((radar_points_tesseract, np.ones(len(radar_points_tesseract))))
                        radar_points = (tesseract2lidar @ radar_points_homo.T).T[:, :3]
                        
                        # 过滤无效点
                        valid_mask = np.isfinite(radar_points).all(axis=1)
                        radar_points = radar_points[valid_mask] if valid_mask.any() else None
                        
                        # 同时过滤速度和反射强度
                        if radar_intensity is not None:
                            radar_intensity = radar_intensity[valid_mask]
                        if radar_velocity is not None:
                            radar_velocity = radar_velocity[valid_mask]
                        
                        # ========== 基于功率的预过滤：保留信号最强的点 ==========
                        # EAsparse 格式每帧约 64,000 点，大量为低功率噪声。
                        # 用 top-1 功率值（power_val[0]）排序，保留 top-K 最强点。
                        max_radar_points_dataset = 10000  # dataset 级上限
                        if (radar_points is not None
                                and power_val is not None
                                and len(radar_points) > max_radar_points_dataset):
                            # power_val[0] = top-1 功率值（log10 scale），越大信号越强
                            top1_power = power_val[0, valid_mask]
                            topk_idx = np.argpartition(top1_power, -max_radar_points_dataset)[-max_radar_points_dataset:]
                            radar_points = radar_points[topk_idx]
                            if radar_intensity is not None:
                                radar_intensity = radar_intensity[topk_idx]
                            if radar_velocity is not None:
                                radar_velocity = radar_velocity[topk_idx]
                            # 同步更新 power_val（后续构建 radar_features 需要）
                            valid_mask_new = np.zeros(len(valid_mask), dtype=bool)
                            valid_indices = np.where(valid_mask)[0]
                            valid_mask_new[valid_indices[topk_idx]] = True
                            valid_mask = valid_mask_new
                        
                        # Model input is exactly: 8D spectrum descriptor + intensity
                        # + raw signed aliased Doppler. Dealiasing happens in VelocityLoss.
                        radar_features = None
                        if power_val is not None:
                            power_val_valid = power_val[:, valid_mask]  # [8, N_valid]
                            top3_idx = power_val_valid[3:6, :]
                            if ((top3_idx < 0) | (top3_idx >= num_doppler_bins)).any():
                                raise ValueError("top-3 Doppler indices must be in [0, 63]")
                            parsed_8 = power_val_valid.T.astype(np.float32)
                            parts = [parsed_8]
                            if radar_intensity is not None:
                                parts.append(radar_intensity.reshape(-1, 1))
                            if radar_velocity is not None:
                                parts.append(radar_velocity.reshape(-1, 1))
                            radar_features = np.concatenate(parts, axis=1)  # [N_valid, 8/9/10]
                except Exception as e:
                    raise RuntimeError(f"failed to load radar data {tesseract_full_path}") from e
        
        input_dict = dict(
            sample_idx=info.get("token", ""),
            occ_path=occ_path_full,  # 使用完整路径
            timestamp=info.get("timestamp"),
            img_filename=image_paths,
            # 不使用lidar，pts_filename设为空字符串
            pts_filename="",  # 不使用lidar数据
            ego2lidar=ego2lidar,
            lidar2img=np.asarray(lidar2img_rts),
            ego2img=np.asarray(ego2image_rts),
            cam_positions=np.asarray(cam_positions),
            focal_positions=np.asarray(focal_positions))
        
        # 添加 radar 点云数据（如果存在）
        if radar_points is not None:
            input_dict['radar_points'] = radar_points  # [N, 3] numpy array
            
            # 添加速度和反射强度（如果存在）
            if 'radar_intensity' in locals() and radar_intensity is not None:
                input_dict['radar_intensity'] = radar_intensity  # [N] numpy array
            if 'radar_velocity' in locals() and radar_velocity is not None:
                input_dict['radar_velocity'] = radar_velocity  # [N] numpy array
            if 'radar_features' in locals() and radar_features is not None:
                input_dict['radar_features'] = radar_features  # [N, F] numpy, F=8 or 10 (power_val + intensity + velocity)

        if input_dict['timestamp'] is None:
            raise KeyError(f"sample {input_dict['sample_idx']} has no timestamp")
        if input_dict.get('radar_points') is None or input_dict.get('radar_features') is None:
            raise ValueError(f"sample {input_dict['sample_idx']} has no valid radar input")
        if input_dict['radar_features'].shape[1] != 10:
            raise ValueError(
                f"sample {input_dict['sample_idx']} radar feature dimension must be 10, "
                f"got {input_dict['radar_features'].shape}"
            )
        return input_dict

    def __len__(self):
        return len(self.keyframes)
