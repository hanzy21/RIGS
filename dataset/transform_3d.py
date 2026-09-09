import os
import torch
import numpy as np
from numpy import random
import mmcv
from PIL import Image
import math
from copy import deepcopy

from . import OPENOCC_TRANSFORMS


@OPENOCC_TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']] # (H, W, C) -> (C, H, W)
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                imgs = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = torch.from_numpy(imgs)
        return results

    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class KRadarAdaptor(object):
    def __init__(self, num_cams, use_ego=False):
        self.num_cams = num_cams
        self.projection_key = 'ego2img' if use_ego else 'lidar2img'

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict[self.projection_key])
        )
        # img_shape 格式: (H, W, C, N)，转换为 image_wh: (N, 2) with [W, H]
        H, W, C, N = input_dict["img_shape"]
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array([(W, H)] * N, dtype=np.float32)
        )
        # 确保 radar_points, radar_intensity, radar_velocity 能正确传递（如果存在）
        if 'radar_points' in input_dict:
            # radar_points 已经是 numpy array，直接保留
            pass
        if 'radar_intensity' in input_dict:
            # radar_intensity 已经是 numpy array，直接保留
            pass
        if 'radar_velocity' in input_dict:
            # radar_velocity 已经是 numpy array，直接保留
            pass
        if 'radar_features' in input_dict:
            # radar_features [N, F] 供 encoder radar 分支
            pass
        return input_dict


@OPENOCC_TRANSFORMS.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_configs = results.get("aug_configs")
        if aug_configs is None:
            return results
        resize, resize_dims, crop, flip, rotate = aug_configs
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            mat = np.eye(4)
            mat[:3, :3] = ida_mat
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            results["ego2img"][i] = mat @ results["ego2img"][i]

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat


@OPENOCC_TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', crop_size=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.crop_size = crop_size

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.crop_size is not None:
            img = img[:self.crop_size[0], :self.crop_size[1]]
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['ori_img'] = deepcopy(img)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointFromFile(object):

    def __init__(self, pc_range, num_pts, use_ego=False):
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts

    def __call__(self, results):
        pts_path = results['pts_filename']
        scan = np.fromfile(pts_path, dtype=np.float32)
        scan = scan.reshape((-1, 5))[:, :4]
        scan[:, 3] = 1.0 # n, 4
        if self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)
        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.2
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]
        
        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan.astype(np.float32)
        
        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPseudoPointFromFile(object):

    def __init__(self, datapath, pc_range, num_pts, is_ego=True, use_ego=False):
        self.datapath = datapath
        self.is_ego = is_ego
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts
        pass

    def __call__(self, results):
        pts_path = os.path.join(self.datapath, f"{results['sample_idx']}.npy")
        scan = np.load(pts_path)
        if self.is_ego and (not self.use_ego):
            ego2lidar = results['ego2lidar']
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = ego2lidar[None, ...] @ scan[..., None] # p, 4, 1
            scan = np.squeeze(scan, axis=-1)

        if (not self.is_ego) and self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)
        
        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.3
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]
        
        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan
        
        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(object):

    def __init__(self, occ_path, semantic=False, use_ego=False, use_sweeps=False, perturb=False):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and (not use_ego)
        self.use_sweeps = use_sweeps
        self.perturb = perturb

        xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        label_file = os.path.join(self.occ_path, results['pts_filename'].split('/')[-1]+'.npy')
        if os.path.exists(label_file):
            label = np.load(label_file)

            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

            mask = new_label != 0

            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        elif self.use_sweeps:
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            mask = new_label != 0
            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        else:
            raise NotImplementedError

        xyz = self.xyz.copy()
        if getattr(self, "perturb", False):
            # xyz[..., :3] = xyz[..., :3] + (np.random.rand(*xyz.shape[:-1], 3) - 0.5) * (0.5 - 1e-3)
            norm_distribution = np.clip(np.random.randn(*xyz.shape[:-1], 3) / 6, -0.5, 0.5)
            xyz[..., :3] = xyz[..., :3] + norm_distribution * 0.49

        if not self.use_ego:
            occ_xyz = xyz[..., :3]
        else:
            ego2lidar = np.linalg.inv(results['ego2lidar']) # 4, 4
            occ_xyz = ego2lidar[None, None, None, ...] @ xyz[..., None] # x, y, z, 4, 1
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        results['occ_xyz'] = occ_xyz
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyKRadar(object):
    """
    K-Radar占用标签加载器
    
    直接使用K-Radar的网格和标签，不进行映射
    
    K-Radar真值参数:
    - voxel_size: 0.4m
    - 空间范围: X=[0, 51.2], Y=[-25.6, 25.6], Z=[-5, 3]
    - 网格大小: 128 x 128 x 20
    - 标签: 9个类别 (0: Empty, 1: Static, 2: Sedan, 3: Bus or Truck, 
                  4: Motorcycle, 5: Bicycle, 6: Pedestrian, 7: Pedestrian Group, 8: Bicycle Group)
    """
    
    # K-Radar标签定义
    KRADAR_CLS_NAMES = [
        'Empty',           # 0
        'Static',          # 1
        'Sedan',           # 2
        'Bus or Truck',    # 3
        'Motorcycle',      # 4
        'Bicycle',         # 5
        'Pedestrian',      # 6
        'Pedestrian Group',# 7
        'Bicycle Group',   # 8
    ]
    
    # Native release labels: empty, background/static, foreground.
    LABEL_REMAP_3CLASS = np.array([0, 1, 2, 2, 2, 2, 2, 2, 2], dtype=np.int64)

    def __init__(self, occ_path, semantic=False, use_ego=False, use_sweeps=False, perturb=False,
                 cc_refine=False, cc_min_component_size=5, cc_morpho_closing=False, cc_closing_radius=1,
                 use_3class=True):
        """
        Args:
            occ_path: 占用标签路径（在K-Radar中不使用，因为occ_path在results中已经是完整路径）
            semantic: 是否使用语义标签
            use_ego: 是否使用ego坐标系
            use_sweeps: 如果没有标签文件，是否创建空标签
            perturb: 是否扰动坐标
            cc_refine: 是否启用连通域 GT 精炼（去噪 + 可选形态学闭合）
            cc_min_component_size: 小于此体素数的连通域视为噪声，设为 empty
            cc_morpho_closing: 是否对动态类做形态学闭合（填充小空洞）
            cc_closing_radius: 闭合操作的结构元素半径（体素数）
        """
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        self.use_sweeps = use_sweeps
        self.perturb = perturb
        self.cc_refine = cc_refine
        self.cc_min_component_size = cc_min_component_size
        self.cc_morpho_closing = cc_morpho_closing
        self.cc_closing_radius = cc_closing_radius
        if not use_3class:
            raise ValueError('the RIGS release supports only native three-class labels')
        self.use_3class = True
        
        # K-Radar网格参数（直接使用，不映射）
        self.voxel_size = 0.4
        self.ranges = [0.0, -25.6, -5.0, 51.2, 25.6, 3.0]  # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.grid = [128, 128, 20]  # 计算得出: (51.2-0)/0.4=128, (25.6-(-25.6))/0.4=128, (3-(-5))/0.4=20
        
        # 创建K-Radar网格
        xyz = self.get_meshgrid(self.ranges, self.grid, self.voxel_size)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)  # x, y, z, 4
        
        # CC Refine 结果缓存：key = label_file 路径，value = refined occ_grid
        self._cc_refine_cache = {}
        
        print(f"[INFO] LoadOccupancyKRadar初始化:")
        print(f"  K-Radar网格: {self.grid}, 范围: {self.ranges}, 体素: {self.voxel_size}m")
        print("  labels: 0=empty, 1=background, 2=foreground")
        if self.cc_refine:
            print(f"  CC精炼: 去噪(min_size={self.cc_min_component_size}), "
                  f"形态学闭合={'ON(r=' + str(self.cc_closing_radius) + ')' if self.cc_morpho_closing else 'OFF'}")

    def get_meshgrid(self, ranges, grid, reso):
        """生成网格坐标"""
        x_min, y_min, z_min, x_max, y_max, z_max = ranges
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + x_min
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + y_min
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + z_min

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([xxx, yyy, zzz], dim=-1).numpy()
        return xyz  # x, y, z, 3

    def _cc_refine_labels(self, occ_grid):
        """连通域 GT 精炼：去除噪声碎片 + 可选形态学闭合填充空洞。
        
        Args:
            occ_grid: [H, W, D] np.int64 OCC GT 标签网格
            
        Returns:
            refined: [H, W, D] np.int64 精炼后的标签网格
        """
        from scipy.ndimage import label as cc_label_func
        from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
        
        refined = occ_grid.copy()
        struct_26 = np.ones((3, 3, 3), dtype=np.int32)  # 26-连通
        
        total_removed = 0
        total_filled = 0
        
        # 对每个非 empty 类别分别处理
        unique_labels = np.unique(occ_grid)
        for cls_id in unique_labels:
            if cls_id == 0:  # 跳过 empty
                continue
            
            cls_mask = (occ_grid == cls_id)
            
            # 1. 连通域分析 + 噪声去除
            labeled_arr, n_comp = cc_label_func(cls_mask, structure=struct_26)
            for comp_id in range(1, n_comp + 1):
                comp_mask = (labeled_arr == comp_id)
                comp_size = comp_mask.sum()
                if comp_size < self.cc_min_component_size:
                    # 小连通域视为噪声，设为 empty
                    refined[comp_mask] = 0
                    total_removed += comp_size
            
            # 2. 形态学闭合（仅对动态类，且仅在启用时）
            if self.cc_morpho_closing and cls_id >= 2:
                # 闭合 = 膨胀 + 腐蚀，填充物体内部小空洞
                r = self.cc_closing_radius
                struct_close = generate_binary_structure(3, 2)  # 18-连通
                # 用精炼后的 mask（已去噪）
                cls_refined = (refined == cls_id)
                closed = binary_dilation(cls_refined, structure=struct_close, iterations=r)
                closed = binary_erosion(closed, structure=struct_close, iterations=r)
                # 只填充新增的体素（膨胀后多出来的 & 原本为 empty 的）
                new_voxels = closed & (~cls_refined) & (refined == 0)
                refined[new_voxels] = cls_id
                total_filled += new_voxels.sum()
        
        if not hasattr(self, '_cc_log_count'):
            self._cc_log_count = 0
        self._cc_log_count += 1
        if self._cc_log_count <= 5:
            print(f"[CC Refine] Removed {total_removed} noise voxels, filled {total_filled} hole voxels")
        
        return refined

    def _load_occ_grid(self, label_file):
        """从 .npy 文件加载 K-Radar OCC 标签网格。

        处理流程与 __call__ 中的当前帧加载完全一致：
        稀疏 (N,4) → [128,128,20] 网格，Y轴翻转，CC精炼。

        Returns:
            np.ndarray [H, W, D] int64 语义标签网格, 或 None (加载失败)
        """
        if not label_file or not os.path.exists(label_file):
            return None
        try:
            label = np.load(label_file)
            if len(label.shape) != 2 or label.shape[1] != 4:
                return None
            coords = label[:, :3].astype(int)
            labels = label[:, 3].astype(int)
            if len(coords) and (
                (coords < 0).any()
                or (coords >= np.asarray(self.grid)).any()
            ):
                raise ValueError(f"occupancy coordinates outside grid {self.grid}")
            if len(labels) and (labels.min() < 0 or labels.max() >= len(self.KRADAR_CLS_NAMES)):
                raise ValueError("occupancy labels must be in [0, 8]")
            if len(coords) > 0:
                coords[:, 1] = self.grid[1] - 1 - coords[:, 1]
            new_label = np.zeros(self.grid, dtype=np.int64)
            if len(coords) > 0:
                new_label[coords[:, 0], coords[:, 1], coords[:, 2]] = labels
            if self.cc_refine and len(coords) > 0:
                if label_file in self._cc_refine_cache:
                    new_label = self._cc_refine_cache[label_file].copy()
                else:
                    new_label = self._cc_refine_labels(new_label)
                    self._cc_refine_cache[label_file] = new_label.copy()
                    if len(self._cc_refine_cache) > 128:
                        oldest_key = next(iter(self._cc_refine_cache))
                        del self._cc_refine_cache[oldest_key]
            return self.LABEL_REMAP_3CLASS[new_label]
        except Exception as exc:
            raise RuntimeError(f"failed to load previous occupancy grid {label_file}") from exc

    def __call__(self, results):
        """加载K-Radar占用标签"""
        label_file = None
        
        # 优先使用occ_path（K-Radar数据集）
        if 'occ_path' in results and results['occ_path']:
            occ_path = results['occ_path']
            if os.path.isabs(occ_path):
                label_file = occ_path
            else:
                label_file = os.path.join(self.occ_path, occ_path)
        
        # 加载标签文件
        if label_file and os.path.exists(label_file):
            try:
                label = np.load(label_file)
                
                # 检查标签格式
                if len(label.shape) != 2 or label.shape[1] != 4:
                    raise ValueError(f"标签文件格式错误: 期望形状(N, 4)，实际形状{label.shape}")
                
                # 提取K-Radar网格坐标和标签
                coords = label[:, :3].astype(int)  # (N, 3)
                labels = label[:, 3].astype(int)   # (N,)
                
                # 检查坐标范围
                if len(coords) and (
                    (coords < 0).any() or (coords >= np.asarray(self.grid)).any()
                ):
                    raise ValueError(
                        f"K-Radar coordinates outside grid {self.grid}: "
                        f"X=[{coords[:, 0].min()}, {coords[:, 0].max()}], "
                        f"Y=[{coords[:, 1].min()}, {coords[:, 1].max()}], "
                        f"Z=[{coords[:, 2].min()}, {coords[:, 2].max()}]")
                
                # 检查标签范围 (0-8)
                if len(labels) and (
                    labels.min() < 0 or labels.max() >= len(self.KRADAR_CLS_NAMES)
                ):
                    raise ValueError(
                        f"K-Radar labels outside [0, {len(self.KRADAR_CLS_NAMES)-1}]: "
                        f"[{labels.min()}, {labels.max()}]")
                
                # ========== 关键修改：翻转Y轴坐标索引以匹配图像坐标系 ==========
                # 图像左边对应Y轴负向，但真值数据可能Y轴方向相反
                # 在填充之前翻转coords的Y坐标索引，这样填充后的数据就已经是翻转后的了
                if len(coords) > 0:
                    coords[:, 1] = self.grid[1] - 1 - coords[:, 1]  # 翻转Y坐标索引
                
                # 创建K-Radar占用网格（直接使用，不映射）
                # 0是Empty类别，1-8是有效类别
                new_label = np.ones(self.grid, dtype=np.int64) * 0  # 初始化为empty（0）
                
                if len(coords) > 0:
                    # 填充占用网格（直接使用K-Radar标签 0-8）
                    # coords的Y坐标索引已经翻转，所以填充后的数据就是翻转后的
                    new_label[coords[:, 0], coords[:, 1], coords[:, 2]] = labels

                new_label = self.LABEL_REMAP_3CLASS[new_label]
                
                # ========== 连通域 GT 精炼（可选，带缓存） ==========
                if self.cc_refine and len(coords) > 0:
                    if label_file in self._cc_refine_cache:
                        new_label = self._cc_refine_cache[label_file].copy()
                    else:
                        new_label = self._cc_refine_labels(new_label)
                        self._cc_refine_cache[label_file] = new_label.copy()
                        # 限制缓存大小（保留最近 64 个样本）
                        if len(self._cc_refine_cache) > 64:
                            oldest_key = next(iter(self._cc_refine_cache))
                            del self._cc_refine_cache[oldest_key]
                
                # ========== 关键修改：mask应该包含所有体素（empty和非empty），以便模型学习预测empty类别 ==========
                # 之前的逻辑（mask = new_label != 0）只标记非empty体素，导致empty体素不参与训练
                # 现在改为全True mask，对所有体素（包括empty）计算loss
                # 如果未来需要根据相机可见性生成mask，可以在这里添加相机FOV的判断逻辑
                mask = np.ones(self.grid, dtype=bool)  # 全True mask，包含所有体素
                # 如果使用语义标签，直接返回标签；否则返回二值标签（0/1）
                if self.semantic:
                    results['occ_label'] = new_label
                else:
                    results['occ_label'] = (new_label != 0).astype(np.int64)
                
                results['occ_cam_mask'] = mask
                
            except Exception as e:
                raise RuntimeError(f"failed to load K-Radar occupancy {label_file}") from e
        elif self.use_sweeps:
            raise FileNotFoundError('empty-label fallback is disabled in the release pipeline')
        else:
            raise FileNotFoundError(
                f"K-Radar occupancy label file not found: {label_file if label_file else 'N/A'}"
            )

        # G1: 加载上一帧 OCC 标签（用于时序位移解混叠）
        prev_occ_path = results.get('prev_occ_path', None)
        if prev_occ_path:
            prev_grid = self._load_occ_grid(prev_occ_path)
            results['prev_occ_label'] = prev_grid  # [H, W, D] or None
        else:
            results['prev_occ_label'] = None

        # 创建占用网格坐标
        xyz = self.xyz.copy()
        if getattr(self, "perturb", False):
            norm_distribution = np.clip(np.random.randn(*xyz.shape[:-1], 3) / 6, -0.5, 0.5)
            xyz[..., :3] = xyz[..., :3] + norm_distribution * 0.49

        if not self.use_ego:
            occ_xyz = xyz[..., :3]
        else:
            ego2lidar = np.linalg.inv(results['ego2lidar'])
            occ_xyz = ego2lidar[None, None, None, ...] @ xyz[..., None]
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        
        # ========== 翻转Y轴坐标以匹配翻转后的网格索引 ==========
        # 翻转Y轴：在axis=1（Y维度）上翻转Y坐标值的顺序
        # 翻转索引后，索引i对应的Y坐标值应该是原来的索引(grid[1]-1-i)对应的Y坐标值
        # 所以我们需要翻转Y坐标值的顺序，使其与翻转后的索引对应
        # 使用.copy()避免负步长问题（PyTorch不支持负步长的numpy数组）
        occ_xyz = np.flip(occ_xyz, axis=1).copy()
        
        results['occ_xyz'] = occ_xyz
        if np.sum(results['occ_cam_mask']) == 0:
            print(f"Warning: No valid occupancy mask found for {results['sample_idx']}")
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(grid={self.grid}, ranges={self.ranges}, voxel_size={self.voxel_size}m)'
        return repr_str
