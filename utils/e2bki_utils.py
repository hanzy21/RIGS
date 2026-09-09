"""
E2-BKI工具函数

提供E2-BKI相关的辅助函数，如sensor_position获取、坐标转换等
"""
import torch
import numpy as np


def get_sensor_position(gaussians, sensor_positions_per_gaussian=None, fallback_to_origin=True):
    """
    获取sensor position用于Gaussian Refinement
    
    IMPORTANT: Each Gaussian should have its own sensor_position (the sensor position
    when that Gaussian was detected), not a fixed position. This allows pruning based
    on relative sensor distances across different frames.
    
    Args:
        gaussians: GaussianPrediction对象或包含means的dict（必须是batched）
        sensor_positions_per_gaussian: [B, G, 3] - 每个Gaussian对应的传感器位置（推荐）
            如果为None，将使用fallback策略
        fallback_to_origin: 如果无法获取sensor position，是否使用原点作为fallback（默认True）
            注意：这只是fallback，实际应该提供每个Gaussian的传感器位置
    
    Returns:
        sensor_position: [B, G, 3] - Sensor position tensor (batched)
    """
    # 如果提供了每个Gaussian的sensor position，直接返回
    if sensor_positions_per_gaussian is not None:
        if len(sensor_positions_per_gaussian.shape) != 3 or sensor_positions_per_gaussian.shape[2] != 3:
            raise ValueError(f"sensor_positions_per_gaussian must be [B, G, 3], got {sensor_positions_per_gaussian.shape}")
        return sensor_positions_per_gaussian
    
    # 提取means以确定形状和device
    if hasattr(gaussians, 'means'):
        means = gaussians.means
    elif isinstance(gaussians, dict) and 'means' in gaussians:
        means = gaussians['means']
    else:
        raise ValueError("gaussians must have 'means' attribute or be a dict with 'means' key")
    
    # 验证必须是batched格式
    if len(means.shape) != 3 or means.shape[2] != 3:
        raise ValueError(f"gaussians.means must be [B, G, 3], got {means.shape}")
    
    # 获取device
    device = means.device
    
    # Fallback: 使用原点（不推荐，但作为fallback）
    # 注意：这假设所有Gaussians在同一帧检测到，或传感器位置相同
    # 实际应该提供每个Gaussian的传感器位置
    B, G = means.shape[:2]
    # 返回[B, G, 3]，每个Gaussian使用相同的原点（fallback）
    sensor_position = torch.zeros(B, G, 3, device=device)
    
    if fallback_to_origin:
        return sensor_position
    else:
        raise ValueError("sensor_positions_per_gaussian must be provided for accurate pruning")


def apply_refinement(refinement, gaussians, sensor_positions=None):
    """
    应用Gaussian Refinement
    
    IMPORTANT: sensor_positions should be the sensor position when each Gaussian was detected,
    not a fixed position. This allows pruning based on relative sensor distances across different frames.
    
    Args:
        refinement: GaussianRefinement对象
        gaussians: GaussianPrediction对象（必须是batched）
        sensor_positions: [B, G, 3] 或 [B, 3] - Sensor position(s) (batched)
            - [B, G, 3]: 每个Gaussian的传感器位置（推荐）
            - [B, 3]: 单个传感器位置（fallback，不推荐）
            如果为None，将使用原点作为fallback
    
    Returns:
        refined_gaussians: 精炼后的GaussianPrediction对象（batched）
    """
    if refinement is None:
        return gaussians
    
    # 如果没有提供sensor_positions，使用原点作为fallback
    if sensor_positions is None:
        sensor_positions = get_sensor_position(gaussians, fallback_to_origin=True)
    
    # 应用refinement
    refined_gaussians = refinement.forward(gaussians, sensor_positions)
    
    return refined_gaussians


def transform_points_to_global(points_local, ego2global):
    """
    将局部坐标点转换为全局坐标
    
    使用齐次变换：P_global = T_ego2global @ P_ego_homogeneous
    
    Args:
        points_local: [B, N, 3] - 局部坐标点（ego坐标系）
        ego2global: [B, 4, 4] 或 [4, 4] - ego到global的变换矩阵
    
    Returns:
        points_global: [B, N, 3] - 全局坐标点
    """
    if points_local is None:
        return None
    
    # 输入验证
    if len(points_local.shape) != 3 or points_local.shape[2] != 3:
        raise ValueError(f"points_local must be [B, N, 3], got {points_local.shape}")
    
    B, N, _ = points_local.shape
    device = points_local.device
    
    # 确保ego2global是tensor
    if isinstance(ego2global, np.ndarray):
        ego2global = torch.from_numpy(ego2global).float()
    
    if not isinstance(ego2global, torch.Tensor):
        raise ValueError(f"ego2global must be torch.Tensor or np.ndarray, got {type(ego2global)}")
    
    # 处理batch维度
    if ego2global.dim() == 2:
        # [4, 4] -> [1, 4, 4] -> [B, 4, 4]
        ego2global = ego2global.unsqueeze(0).expand(B, -1, -1)
    elif ego2global.dim() == 3:
        # [B, 4, 4]
        if ego2global.shape[0] != B:
            raise ValueError(f"ego2global batch size {ego2global.shape[0]} != points_local batch size {B}")
    else:
        raise ValueError(f"ego2global must be [B, 4, 4] or [4, 4], got {ego2global.shape}")
    
    # 确保在正确的device上
    if ego2global.device != device:
        ego2global = ego2global.to(device)
    
    # 验证矩阵形状
    if ego2global.shape[1:] != (4, 4):
        raise ValueError(f"ego2global must be 4x4 matrix, got {ego2global.shape[1:]}")
    
    # 转换为齐次坐标 [B, N, 4]
    ones = torch.ones(B, N, 1, device=device, dtype=points_local.dtype)
    points_homogeneous = torch.cat([points_local, ones], dim=-1)  # [B, N, 4]
    
    # 应用变换矩阵: [B, N, 4] @ [B, 4, 4]^T = [B, N, 4]
    points_global_homogeneous = torch.matmul(points_homogeneous, ego2global.transpose(-1, -2))
    
    # 提取3D坐标 [B, N, 3]
    points_global = points_global_homogeneous[..., :3]
    
    return points_global


def get_ego2global_from_metas(metas, device=None, verbose=False):
    """
    从metas中获取ego2global变换矩阵（增强版：支持多种数据结构和位置）
    
    Args:
        metas: dict, list, or any structure - 包含curr_ego2global的数据结构
            支持的查找位置（按优先级）:
            1. metas['curr_ego2global'] (直接位于顶层，最常见)
            2. metas['metas']['curr_ego2global'] (嵌套在metas子字典中)
            3. 如果metas是list，遍历每个元素查找
        device: torch.device - 目标设备（可选）
        verbose: bool - 是否打印调试信息（默认False）
    
    Returns:
        ego2global: [B, 4, 4] - 变换矩阵（tensor），如果无法获取则返回None
    """
    if metas is None:
        if verbose:
            print("[get_ego2global_from_metas] Warning: metas is None")
        return None
    
    def _extract_ego2global_from_dict(d, location="", verbose=False):
        """
        从单个dict中提取ego2global
        
        优化逻辑：找到后立即返回，不再查找其他位置
        - 优先在顶层查找 curr_ego2global
        - 如果找到，立即返回（不再查找嵌套位置）
        - 只有找不到时，才尝试在嵌套的metas中查找
        """
        if not isinstance(d, dict):
            return None
        
        # 方法1: 直接在顶层查找（最优先，找到后立即返回，不再查找其他位置）
        if 'curr_ego2global' in d and d['curr_ego2global'] is not None:
            ego2global = d['curr_ego2global']
            if verbose:
                print(f"[get_ego2global_from_metas] Found curr_ego2global at {location}/curr_ego2global")
            
            # 转换类型
            if isinstance(ego2global, np.ndarray):
                ego2global = torch.from_numpy(ego2global).float()
            elif isinstance(ego2global, torch.Tensor):
                ego2global = ego2global.float()
            else:
                if verbose:
                    print(f"[get_ego2global_from_metas] Warning: curr_ego2global has unsupported type: {type(ego2global)}")
                return None
            
            # 验证和规范化形状
            if ego2global.shape == (4, 4):
                ego2global = ego2global.unsqueeze(0)  # [1, 4, 4]
            elif ego2global.shape == (1, 4, 4):
                pass  # 已经是 [1, 4, 4]
            elif len(ego2global.shape) == 3 and ego2global.shape[0] == 1:
                # 可能是 [1, 4, 4] 但需要验证
                if ego2global.shape[1:] == (4, 4):
                    pass
                else:
                    if verbose:
                        print(f"[get_ego2global_from_metas] Warning: curr_ego2global has unexpected shape: {ego2global.shape}")
                    return None
            else:
                if verbose:
                    print(f"[get_ego2global_from_metas] Warning: curr_ego2global has unexpected shape: {ego2global.shape}")
                return None
            
            return ego2global
        
        # 方法2: 在嵌套的metas中查找
        if 'metas' in d and isinstance(d['metas'], dict):
            nested_result = _extract_ego2global_from_dict(d['metas'], location=location + "/metas", verbose=verbose)
            if nested_result is not None:
                return nested_result
        
        return None
    
    # 处理list类型（batch情况）
    if isinstance(metas, list):
        ego2global_list = []
        for idx, meta in enumerate(metas):
            ego2global = _extract_ego2global_from_dict(meta, location=f"list[{idx}]", verbose=verbose)
            if ego2global is not None:
                # 确保是 [1, 4, 4] 格式
                if ego2global.shape == (4, 4):
                    ego2global = ego2global.unsqueeze(0)
                ego2global_list.append(ego2global.squeeze(0))  # 去掉batch维度以便stack
        
        if len(ego2global_list) > 0:
            ego2global_tensor = torch.stack(ego2global_list)  # [B, 4, 4]
            if device is not None:
                ego2global_tensor = ego2global_tensor.to(device)
            if verbose:
                print(f"[get_ego2global_from_metas] Successfully extracted {len(ego2global_list)} ego2global matrices, shape: {ego2global_tensor.shape}")
            return ego2global_tensor
    
    # 处理dict类型（单个样本）
    elif isinstance(metas, dict):
        ego2global = _extract_ego2global_from_dict(metas, location="metas", verbose=verbose)
        if ego2global is not None:
            # 确保是 [1, 4, 4] 格式
            if ego2global.shape == (4, 4):
                ego2global = ego2global.unsqueeze(0)
            elif ego2global.shape[0] != 1 and len(ego2global.shape) == 3:
                # 如果已经是 [B, 4, 4]，保持不变
                pass
            
            if device is not None:
                ego2global = ego2global.to(device)
            
            if verbose:
                print(f"[get_ego2global_from_metas] Successfully extracted ego2global, shape: {ego2global.shape}")
            return ego2global
    
    if verbose:
        print(f"[get_ego2global_from_metas] Warning: Could not find curr_ego2global in metas (type: {type(metas)})")
        if isinstance(metas, dict):
            print(f"  Available keys: {list(metas.keys())[:20]}")  # 只显示前20个key避免输出过长
    
    return None


def prepare_query_points_from_occupancy(metas, gaussian_obj, cfg, device='cuda'):
    """
    从occupancy grid中准备query points，可选择只使用非empty的点
    
    Args:
        metas: dict - 包含occ_xyz和occ_label的字典
        gaussian_obj: GaussianPrediction对象，用于确定batch size
        cfg: config对象，包含e2bki_config
        device: torch.device - 目标设备
    
    Returns:
        query_points: [B, N, 3] - Query points tensor（只包含非empty点，如果启用）
    """
    import numpy as np
    
    e2bki_config = cfg.e2bki_config if hasattr(cfg, 'e2bki_config') else {}
    use_non_empty_only = e2bki_config.get('use_non_empty_only', False)
    empty_label = e2bki_config.get('empty_label', 0)
    
    # 获取occupancy grid配置
    if hasattr(cfg, 'model') and 'lifter' in cfg.model:
        lifter_cfg = cfg.model['lifter']
        occ_resolution = lifter_cfg.get('occ_resolution', [128, 128, 20])
        pc_range = lifter_cfg.get('pc_range', [0.0, -25.6, -5.0, 51.2, 25.6, 3.0])
    else:
        occ_resolution = [128, 128, 20]
        pc_range = [0.0, -25.6, -5.0, 51.2, 25.6, 3.0]
    
    H, W, D = occ_resolution
    B = gaussian_obj.means.shape[0] if len(gaussian_obj.means.shape) == 3 else 1
    
    # 方法1: 尝试从metas获取occ_xyz和occ_label
    occ_xyz = None
    occ_label = None
    
    if isinstance(metas, dict):
        if 'occ_xyz' in metas and metas['occ_xyz'] is not None:
            occ_xyz = metas['occ_xyz']
            if isinstance(occ_xyz, np.ndarray):
                occ_xyz = torch.from_numpy(occ_xyz).float()
        if 'occ_label' in metas and metas['occ_label'] is not None:
            occ_label = metas['occ_label']
            if isinstance(occ_label, np.ndarray):
                occ_label = torch.from_numpy(occ_label)
    
    # 如果成功获取了occ_xyz和occ_label，使用它们
    if occ_xyz is not None:
        # 处理不同的形状
        if isinstance(occ_xyz, torch.Tensor):
            if len(occ_xyz.shape) == 4:  # [H, W, D, 3]
                occ_xyz = occ_xyz.to(device)
                occ_xyz_flat = occ_xyz.reshape(-1, 3)  # [H*W*D, 3]
                occ_xyz_batch = occ_xyz_flat.unsqueeze(0).expand(B, -1, -1)  # [B, H*W*D, 3]
            elif len(occ_xyz.shape) == 5:  # [B, H, W, D, 3]
                occ_xyz = occ_xyz.to(device)
                if occ_xyz.shape[0] == 1:
                    occ_xyz_batch = occ_xyz[0].reshape(-1, 3).unsqueeze(0).expand(B, -1, -1)  # [B, H*W*D, 3]
                else:
                    occ_xyz_batch = occ_xyz.reshape(B, -1, 3)  # [B, H*W*D, 3]
            else:
                occ_xyz_batch = None
        else:
            occ_xyz_batch = None
    else:
        occ_xyz_batch = None
    
    # 处理occ_label
    if occ_label is not None and isinstance(occ_label, torch.Tensor):
        occ_label = occ_label.to(device)
        if len(occ_label.shape) == 3:  # [H, W, D]
            occ_label_flat = occ_label.reshape(-1)  # [H*W*D]
        elif len(occ_label.shape) == 4:  # [B, H, W, D]
            if occ_label.shape[0] == 1:
                occ_label_flat = occ_label[0].reshape(-1)  # [H*W*D]
            else:
                occ_label_flat = occ_label.reshape(B, -1)[0]  # 使用第一个batch的label [H*W*D]
        else:
            occ_label_flat = None
    else:
        occ_label_flat = None
    
    # 如果启用了use_non_empty_only且有occ_label，筛选非empty点
    if use_non_empty_only and occ_label_flat is not None and occ_xyz_batch is not None:
        # 创建非empty mask
        non_empty_mask = (occ_label_flat != empty_label)  # [H*W*D]
        
        if non_empty_mask.any():
            # 筛选非empty的点
            # occ_xyz_batch是[B, H*W*D, 3]，我们需要对每个batch应用相同的mask
            # 取第一个batch的点，应用mask，然后expand到所有batch
            query_points_flat = occ_xyz_batch[0]  # [H*W*D, 3]
            query_points_filtered = query_points_flat[non_empty_mask]  # [N_non_empty, 3]
            
            # Expand到batch维度
            N_filtered = query_points_filtered.shape[0]
            query_points = query_points_filtered.unsqueeze(0).expand(B, -1, -1)  # [B, N_non_empty, 3]
            
            num_total = occ_label_flat.shape[0]
            num_non_empty = non_empty_mask.sum().item()
            print(f"  - Filtered query points: {num_non_empty}/{num_total} non-empty points "
                  f"({num_non_empty/num_total*100:.1f}%)")
            
            return query_points
        else:
            print(f"  - Warning: No non-empty points found! Using all points.")
    
    # 如果没有occ_xyz，生成完整的grid
    if occ_xyz_batch is None:
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        
        x = torch.linspace(x_min, x_max, W, device=device)
        y = torch.linspace(y_min, y_max, H, device=device)
        z = torch.linspace(z_min, z_max, D, device=device)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        query_points_flat = torch.stack([xx, yy, zz], dim=-1)  # [W, H, D, 3]
        query_points_flat = query_points_flat.permute(1, 0, 2, 3)  # [H, W, D, 3]
        query_points_flat = query_points_flat.reshape(-1, 3)  # [H*W*D, 3]
        
        # 如果有occ_label且启用了筛选，应用筛选
        if use_non_empty_only and occ_label_flat is not None:
            non_empty_mask = (occ_label_flat != empty_label)
            if non_empty_mask.any():
                query_points_flat = query_points_flat[non_empty_mask]
                num_total = H * W * D
                num_non_empty = non_empty_mask.sum().item()
                print(f"  - Generated grid and filtered: {num_non_empty}/{num_total} non-empty points "
                      f"({num_non_empty/num_total*100:.1f}%)")
        
        query_points = query_points_flat.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]
        return query_points
    
    # 如果没有启用筛选，直接返回所有点
    return occ_xyz_batch

