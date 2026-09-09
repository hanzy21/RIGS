"""
Anisotropic Kernel for E2-BKI

This module implements anisotropic kernel computation based on Gaussian covariance matrices.
The kernel adapts to local geometry, enabling geometry-aligned spatial propagation.

Reference:
    Kim et al., "E2-BKI: Evidential Ellipsoidal Bayesian Kernel Inference", 2025
    Section IV-D: Evidential Ellipsoidal BKI

Implemented Features:
    - Equation (2): Base kernel function k'(d, ℓ) with cos/sin terms
    - Equation (7): Ellipsoid surface distance computation (approximate)
    - Covariance matrix computation: Cov = R * S * S^T * R^T
    - Support for uncertainty-adaptive kernel radius (used in EvidentialEllipsoidalBKI)
"""

import torch
import torch.nn as nn
from ..utils.utils import get_rotation_matrix
from .timing_utils import TimingProfiler


class AnisotropicKernel(nn.Module):
    """
    Anisotropic Kernel for geometry-aligned spatial propagation.
    
    The kernel is based on Gaussian covariance matrices:
        Cov = R * S * S^T * R^T
        K(x, μ) = exp(-0.5 * (x - μ)^T * Cov^(-1) * (x - μ))
    
    where:
        - R: rotation matrix (from quaternion) [3×3]
        - S: scale diagonal matrix [3×3], S = diag(scale_x, scale_y, scale_z)
        - x: query point [3]
        - μ: Gaussian center [3]
    
    Args:
        scale (float): Kernel scale factor (default: 1.0)
        eps (float): Small epsilon for numerical stability (default: 1e-8)
        max_mahalanobis_dist (float): Maximum Mahalanobis distance to avoid exp overflow (default: 50.0)
    """
    
    def __init__(self, scale=1.0, eps=1e-8, max_mahalanobis_dist=50.0, verbose=False, 
                 use_paper_kernel=True, max_euclidean_distance=5.0, use_streaming_chunking=True,
                 use_blockwise_pruning=True, block_size=4096, blockwise_preselect_gaussians=128):
        """
        Args:
            scale (float): Kernel scale factor ℓ (default: 1.0)
            eps (float): Small epsilon for numerical stability (default: 1e-8)
            max_mahalanobis_dist (float): Maximum Mahalanobis distance to avoid exp overflow (default: 50.0)
            verbose (bool): Whether to print debug information
            use_paper_kernel (bool): Whether to use paper's k' function (Equation 2) instead of Gaussian kernel
            max_euclidean_distance (float): Maximum Euclidean distance for kernel computation (default: 5.0m)
                Only compute kernel for Gaussians within this distance from query points
            use_streaming_chunking (bool): Whether to use streaming chunking to avoid storing full distance_mask
                in memory (default: True). When enabled, processes query points in chunks and accumulates
                kernel values without storing intermediate distance masks.
            use_blockwise_pruning (bool): Whether to use blockwise spatial pruning (default: True).
                When enabled, groups query points into blocks, computes bounding boxes, and only computes
                kernels for Gaussians within the bounding box. This can significantly reduce computation
                when query points are sparse and Gaussians are dense.
            block_size (int): Size of query point blocks for spatial pruning (default: 4096).
                Larger blocks increase parallelism but may reduce pruning effectiveness.
            blockwise_preselect_gaussians (int or None): Optional per-block Gaussian preselection.
                If set, keep only top-k Gaussians nearest to block center before Mahalanobis
                computation. This is an approximation to reduce kernel runtime.
        """
        super().__init__()
        self.scale = scale  # ℓ in the paper
        self.eps = eps
        self.max_mahalanobis_dist = max_mahalanobis_dist
        self.verbose = verbose
        self.use_paper_kernel = use_paper_kernel
        self.max_euclidean_distance = max_euclidean_distance  # Maximum Euclidean distance for kernel computation
        self.use_streaming_chunking = use_streaming_chunking  # Enable streaming chunking to avoid OOM
        self.use_blockwise_pruning = use_blockwise_pruning  # Enable blockwise spatial pruning
        self.block_size = block_size  # Block size for spatial pruning
        self.blockwise_preselect_gaussians = blockwise_preselect_gaussians
        
        # Timing profiler (shared with parent module)
        self.timing_profiler = None  # Will be set by parent module
    
    def _compute_sparse_kernel_blockwise(self, query_points, gaussian_means, gaussian_covariances,
                                         ell, use_ellipsoid_distance, tau,
                                         sparsify_on_the_fly, sparsify_threshold, sparsify_max_kernels_per_point,
                                         uncertainty_mask, semantics=None, gaussian_uncertainties=None,
                                         aggregation_mode='standard', cc_norm_min_count=1,
                                         base_probs=None, attention_tau=0.3):
        """
        [优化] 分块空间剪枝核计算
        
        原理：将 Query Points 分块，计算每个块的包围盒，只对包围盒内的 Gaussians 进行核函数计算。
        这可以显著减少计算量，特别是当 Query Points 分布稀疏而 Gaussians 密集时。
        
        Args:
            query_points: [B, N, 3] - Query point coordinates
            gaussian_means: [B, G, 3] - Gaussian centers
            gaussian_covariances: [B, G, 3, 3] - Gaussian covariance matrices
            ell: [B, G] or scalar - Kernel scale
            use_ellipsoid_distance: bool - Whether to use ellipsoid distance
            tau: float - Ellipsoid size parameter
            sparsify_on_the_fly: bool - Whether to sparsify on the fly
            sparsify_threshold: float - Sparsification threshold
            sparsify_max_kernels_per_point: int or None - Max kernels per point
            uncertainty_mask: [B, G] or None - Uncertainty mask
            semantics: [B, G, C] or None - Gaussian semantics (for direct aggregation)
            gaussian_uncertainties: [B, G] or None - Gaussian uncertainties (for direct aggregation)
        
        Returns:
            If semantics and gaussian_uncertainties are provided:
                dict with 'alpha_increment' [B, N, C] and 'uncertainty_increment' [B, N]
            Otherwise:
                kernel_values: [B, N, G] (dense) or dict (sparse) - Kernel values
        """
        import time
        B, N, _ = query_points.shape
        _, G, _ = gaussian_means.shape
        device = query_points.device
        
        # Performance profiling
        perf_times = {
            'cov_inv_compute': 0.0,
            'ell_prepare': 0.0,
            'bounding_box': 0.0,
            'spatial_pruning': 0.0,
            'data_extraction': 0.0,
            'euclidean_dist': 0.0,
            'mahalanobis_dist': 0.0,
            'ellipsoid_dist': 0.0,
            'kernel_compute': 0.0,
            'sparsify': 0.0,
            'sparsify_threshold_mask': 0.0,  # 阈值过滤
            'sparsify_topk': 0.0,  # TopK 选择
            'sparsify_index_mapping': 0.0,  # 索引映射
            'sparsify_data_extraction': 0.0,  # 数据提取（逐点循环）
            'sparsify_list_append': 0.0,  # 列表追加
            'index_mapping': 0.0,
            'memory_ops': 0.0,
        }
        perf_counts = {
            'num_blocks': 0,
            'num_empty_blocks': 0,
            'total_valid_gaussians': 0,
            'total_filtered_gaussians': 0,
        }
        
        # Pre-compute covariance inverse (reused for all blocks)
        t_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eye = torch.eye(3, dtype=gaussian_covariances.dtype, device=device)
        cov_reg = gaussian_covariances + self.eps * eye.view(1, 1, 3, 3)  # [B, G, 3, 3]
        try:
            cov_inv = torch.inverse(cov_reg)  # [B, G, 3, 3]
        except RuntimeError:
            if self.verbose:
                dets = torch.det(cov_reg)
                singular_mask = torch.abs(dets) < 1e-6
                if singular_mask.any():
                    num_singular = singular_mask.sum().item()
                    print(f"[AnisotropicKernel] Warning: {num_singular} singular covariance matrices detected")
            cov_inv = torch.pinverse(cov_reg)  # [B, G, 3, 3]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        perf_times['cov_inv_compute'] = time.time() - t_start
        
        # Ensure ell is [B, G]
        t_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if ell is None:
            ell = self.scale
        if not isinstance(ell, torch.Tensor):
            ell = torch.tensor(ell, dtype=query_points.dtype, device=device)
        if ell.dim() == 0:
            ell = ell.expand(B, G)
        elif ell.dim() == 1:
            if ell.shape[0] == G:
                ell = ell.unsqueeze(0).expand(B, G)
            elif ell.shape[0] == B:
                ell = ell.unsqueeze(1).expand(B, G)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        perf_times['ell_prepare'] = time.time() - t_start
        
        # Initialize output
        # If semantics and uncertainties are provided, directly aggregate alpha and uncertainty
        direct_aggregation = (semantics is not None and gaussian_uncertainties is not None)
        
        if direct_aggregation:
            # Direct aggregation mode: accumulate alpha and uncertainty directly
            _, _, C = semantics.shape
            alpha_accum = torch.zeros(B, N, C, device=device, dtype=query_points.dtype)
            uncertainty_accum = torch.zeros(B, N, device=device, dtype=query_points.dtype)
        elif sparsify_on_the_fly:
            kernel_values_list = [[] for _ in range(B)]
            gaussian_indices_list = [[] for _ in range(B)]
            num_nonzero = torch.zeros(B, N, device=device, dtype=torch.long)
        else:
            kernel_values = torch.zeros(B, N, G, device=device, dtype=query_points.dtype)
        
        # Search radius: must cover max_euclidean_distance + some margin for anisotropic kernels
        # Use a larger margin to ensure we don't miss Gaussians near block boundaries
        # The margin accounts for:
        # 1. Anisotropic kernels can extend beyond Euclidean distance
        # 2. Query points near block boundaries need coverage from adjacent blocks
        # CRITICAL: The bounding box must be large enough to include ALL Gaussians that are
        # within max_euclidean_distance of ANY query point in the block.
        # If a query point is at the edge of the block, its max_euclidean_distance range
        # extends beyond the block's min/max by up to max_euclidean_distance.
        # So we need: search_radius >= max_euclidean_distance to ensure coverage.
        # Using 2.0x provides additional safety margin for numerical precision.
        base_radius = self.max_euclidean_distance if self.max_euclidean_distance is not None else 5.0
        search_radius = base_radius * 2.0  # Use 2.0x margin to ensure no Gaussians are missed
        total_filtered = 0
        total_gaussians_processed = 0
        
            # Process each batch independently
        for b in range(B):
            q_batch = query_points[b]  # [N, 3]
            m_batch = gaussian_means[b]  # [G, 3]
            c_batch = gaussian_covariances[b]  # [G, 3, 3]
            c_inv_batch = cov_inv[b]  # [G, 3, 3]
            ell_batch = ell[b]  # [G]
            u_mask_batch = uncertainty_mask[b] if uncertainty_mask is not None else None  # [G]
            
            # Extract semantics and uncertainties for direct aggregation
            if direct_aggregation:
                w_batch = semantics[b]  # [G, C]
                u_batch = gaussian_uncertainties[b]  # [G]
            
            # Process query points in blocks
            for block_start in range(0, N, self.block_size):
                block_end = min(block_start + self.block_size, N)
                q_block = q_batch[block_start:block_end]  # [block_N, 3]
                block_N = q_block.shape[0]
                perf_counts['num_blocks'] += 1
                
                # Step 1: Compute bounding box for this query block
                t_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                # Expand by search_radius to ensure we capture all relevant Gaussians
                # CRITICAL FIX: Use a larger margin to ensure we don't miss Gaussians that are
                # within max_euclidean_distance of any query point in the block
                # The issue: if a query point is near the edge of the block, its 5m range
                # might extend beyond the block's bounding box even with 1.5x margin
                # Solution: Use 2.0x margin to be safe, or better yet, ensure the bounding box
                # covers max_euclidean_distance from the farthest points in the block
                q_min = q_block.min(dim=0)[0] - search_radius  # [3]
                q_max = q_block.max(dim=0)[0] + search_radius  # [3]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                perf_times['bounding_box'] += time.time() - t_start
                
                # Step 2: Fast spatial pruning: find Gaussians within bounding box
                t_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                # Use vectorized comparison (very fast on GPU)
                in_box_mask = (
                    (m_batch[:, 0] >= q_min[0]) & (m_batch[:, 0] <= q_max[0]) &
                    (m_batch[:, 1] >= q_min[1]) & (m_batch[:, 1] <= q_max[1]) &
                    (m_batch[:, 2] >= q_min[2]) & (m_batch[:, 2] <= q_max[2])
                )  # [G]
                
                # Get valid Gaussian indices
                valid_g_idx = torch.nonzero(in_box_mask, as_tuple=False).squeeze(-1)  # [K]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                perf_times['spatial_pruning'] += time.time() - t_start
                
                if valid_g_idx.numel() == 0:
                    # No Gaussians in this block's bounding box
                    # In direct aggregation mode, alpha_accum and uncertainty_accum remain 0 (already initialized)
                    # In sparse format, we need to add empty tensors
                    perf_counts['num_empty_blocks'] += 1
                    if direct_aggregation:
                        # For direct aggregation, alpha and uncertainty remain 0 (correct behavior)
                        # No need to do anything, just continue
                        pass
                    elif sparsify_on_the_fly:
                        for n_offset in range(block_N):
                            n = block_start + n_offset
                            kernel_values_list[b].append(torch.tensor([], device=device, dtype=query_points.dtype))
                            gaussian_indices_list[b].append(torch.tensor([], device=device, dtype=torch.long))
                            num_nonzero[b, n] = 0
                    continue

                # Optional approximation: preselect top-k Gaussians per block by distance to block center.
                # This bounds Mahalanobis complexity from O(block_N * K) to O(block_N * K_pre).
                pre_k = self.blockwise_preselect_gaussians
                if pre_k is not None and pre_k > 0 and valid_g_idx.numel() > pre_k:
                    block_center = q_block.mean(dim=0, keepdim=True)  # [1, 3]
                    cand_means = m_batch[valid_g_idx]  # [K, 3]
                    center_dist = torch.norm(cand_means - block_center, dim=-1)  # [K]
                    _, keep_local_idx = torch.topk(center_dist, k=pre_k, largest=False)
                    valid_g_idx = valid_g_idx[keep_local_idx]

                K = valid_g_idx.shape[0]  # Number of valid Gaussians
                total_gaussians_processed += K * block_N
                perf_counts['total_valid_gaussians'] += K * block_N
                
                # Step 3: Extract subset data
                t_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sub_means = m_batch[valid_g_idx].unsqueeze(0)  # [1, K, 3]
                sub_covs = c_batch[valid_g_idx].unsqueeze(0)  # [1, K, 3, 3]
                sub_cov_inv = c_inv_batch[valid_g_idx].unsqueeze(0)  # [1, K, 3, 3]
                sub_ell = ell_batch[valid_g_idx].unsqueeze(0)  # [1, K]
                sub_u_mask = u_mask_batch[valid_g_idx].unsqueeze(0) if u_mask_batch is not None else None  # [1, K]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                perf_times['data_extraction'] += time.time() - t_start
                
                # Step 4: Compute Euclidean distances for this block (only for valid Gaussians)
                t_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                q_block_expanded = q_block.unsqueeze(1)  # [block_N, 1, 3]
                sub_means_2d = sub_means.squeeze(0)  # [K, 3]
                block_diff = q_block_expanded - sub_means_2d.unsqueeze(0)  # [block_N, K, 3]
                block_euclidean_distances = torch.norm(block_diff, dim=-1)  # [block_N, K]
                
                # Apply max_euclidean_distance filter
                if self.max_euclidean_distance is not None and self.max_euclidean_distance > 0:
                    block_mask = block_euclidean_distances <= self.max_euclidean_distance  # [block_N, K]
                    total_filtered += block_mask.sum().item()
                    perf_counts['total_filtered_gaussians'] += block_mask.sum().item()
                else:
                    block_mask = torch.ones_like(block_euclidean_distances, dtype=torch.bool)  # [block_N, K]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                perf_times['euclidean_dist'] += time.time() - t_start
                
                # Step 5: Compute distances (Mahalanobis or ellipsoid) for valid Gaussians
                if use_ellipsoid_distance:
                    t_start = time.time()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    # Use ellipsoid surface distance
                    block_ellipsoid_distances = self.compute_ellipsoid_surface_distance(
                        q_block.unsqueeze(0), sub_means, sub_covs, tau=tau
                    )  # [1, block_N, K]
                    block_ellipsoid_distances = block_ellipsoid_distances.squeeze(0)  # [block_N, K]
                    block_distances = torch.where(block_mask, block_ellipsoid_distances,
                                                  torch.tensor(float('inf'), device=device, dtype=query_points.dtype))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    perf_times['ellipsoid_dist'] += time.time() - t_start
                    del block_ellipsoid_distances
                else:
                    t_start = time.time()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    # Compute Mahalanobis distance for valid Gaussians only
                    # block_diff: [block_N, K, 3]
                    # sub_cov_inv: [1, K, 3, 3] -> [K, 3, 3]
                    sub_cov_inv_2d = sub_cov_inv.squeeze(0)  # [K, 3, 3]
                    block_diff_expanded = block_diff.unsqueeze(-1)  # [block_N, K, 3, 1]
                    # Use batched matrix multiplication for efficiency
                    # block_diff: [block_N, K, 3], sub_cov_inv_2d: [K, 3, 3]
                    # We need to compute: block_diff @ sub_cov_inv_2d for each K
                    # Reshape for bmm: [block_N, 1, K, 3] @ [1, K, 3, 3] -> [block_N, K, 3]
                    block_diff_reshaped = block_diff.unsqueeze(2)  # [block_N, K, 1, 3]
                    sub_cov_inv_expanded = sub_cov_inv_2d.unsqueeze(0)  # [1, K, 3, 3]
                    # Use einsum for clarity: nki (block_N, K, 3) @ kij (K, 3, 3) -> nkj (block_N, K, 3)
                    cov_inv_diff = torch.einsum('nki,kij->nkj', block_diff, sub_cov_inv_2d)  # [block_N, K, 3]
                    block_mahalanobis_dist_sq = torch.sum(block_diff * cov_inv_diff, dim=-1)  # [block_N, K]
                    block_mahalanobis_dist_sq = torch.clamp(block_mahalanobis_dist_sq, min=0.0, max=self.max_mahalanobis_dist ** 2)
                    block_mahalanobis_dist = torch.sqrt(block_mahalanobis_dist_sq + self.eps)  # [block_N, K]
                    
                    # Apply distance mask
                    block_distances = torch.where(block_mask, block_mahalanobis_dist,
                                                 torch.tensor(float('inf'), device=device, dtype=query_points.dtype))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    perf_times['mahalanobis_dist'] += time.time() - t_start
                    
                    del block_diff_expanded, cov_inv_diff, block_mahalanobis_dist_sq, block_mahalanobis_dist, sub_cov_inv_2d
                
                # Step 6: Compute kernel values
                t_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                block_distances_safe = torch.where(torch.isinf(block_distances),
                                                   torch.tensor(1e10, device=device, dtype=block_distances.dtype),
                                                   block_distances)
                
                # sub_ell: [1, K] -> expand to [block_N, K] for broadcasting
                sub_ell_2d = sub_ell.squeeze(0)  # [K]
                if self.use_paper_kernel:
                    # k_prime can handle broadcasting: [block_N, K] and [K] -> [block_N, K]
                    block_kernel_values = self.k_prime(block_distances_safe, sub_ell_2d)  # [block_N, K]
                else:
                    block_dist_sq = block_distances_safe ** 2
                    block_dist_sq = torch.clamp(block_dist_sq, min=0.0, max=self.max_mahalanobis_dist ** 2)
                    block_kernel_values = torch.exp(-0.5 * block_dist_sq) * sub_ell_2d.unsqueeze(0)  # [block_N, K]
                    del block_dist_sq
                
                # Apply distance mask
                block_kernel_values = block_kernel_values * block_mask.float()  # [block_N, K]
                
                # Apply uncertainty mask if provided
                if sub_u_mask is not None:
                    sub_u_mask_expanded = sub_u_mask.unsqueeze(0).float()  # [1, 1, K]
                    block_kernel_values = block_kernel_values * sub_u_mask_expanded.squeeze(0)  # [block_N, K]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                perf_times['kernel_compute'] += time.time() - t_start
                
                t_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                del block_distances_safe, block_distances, block_mask, block_euclidean_distances, block_diff
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                perf_times['memory_ops'] += time.time() - t_start
                
                # Step 7: Map back to full Gaussian space and accumulate
                # OPTIMIZED: Direct aggregation or vectorized sparsification
                t_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                if direct_aggregation:
                    # DIRECT AGGREGATION MODE: Aggregate alpha and uncertainty directly on GPU
                    # This avoids Python list operations and CPU-GPU transfers
                    
                    # 1. Threshold filtering (vectorized on GPU)
                    t_sub = time.time()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    # Apply threshold: set values below threshold to 0
                    block_kernel_values_thresholded = torch.where(
                        block_kernel_values > sparsify_threshold,
                        block_kernel_values,
                        torch.tensor(0.0, device=device, dtype=block_kernel_values.dtype)
                    )  # [block_N, K]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    perf_times['sparsify_threshold_mask'] += time.time() - t_sub
                    
                    # 2. Batch Top-K (if max_kernels_per_point is set)
                    if sparsify_max_kernels_per_point is not None:
                        t_sub = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        max_k = min(sparsify_max_kernels_per_point, K)
                        
                        # Replace values below threshold with -inf so they won't be selected by topk
                        # This ensures we only select values above threshold
                        block_kernel_masked = torch.where(
                            block_kernel_values > sparsify_threshold,
                            block_kernel_values,
                            torch.tensor(float('-inf'), device=device, dtype=block_kernel_values.dtype)
                        )  # [block_N, K]
                        
                        topk_values, topk_local_indices = torch.topk(
                            block_kernel_masked,
                            k=max_k,
                            dim=1,  # Top-k along K dimension for each point
                            largest=True
                        )  # [block_N, max_k]
                        
                        # Filter out values that were below threshold (now -inf)
                        valid_topk_mask = topk_values > sparsify_threshold  # [block_N, max_k]
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        perf_times['sparsify_topk'] += time.time() - t_sub
                        
                        # 3. Map to global Gaussian indices
                        t_sub = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        # Use advanced indexing: valid_g_idx[topk_local_indices]
                        # topk_local_indices: [block_N, max_k] with values in [0, K-1]
                        # valid_g_idx: [K] with global Gaussian indices
                        topk_global_indices = valid_g_idx[topk_local_indices]  # [block_N, max_k]
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        perf_times['sparsify_index_mapping'] += time.time() - t_sub
                        
                        # 4. Direct aggregation: compute alpha and uncertainty increment
                        t_sub = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        # Extract local evidence: [K, C]
                        local_evidence = w_batch[valid_g_idx]  # [K, C]
                        # Gather evidence for top-k: [block_N, max_k, C]
                        gathered_evidence = local_evidence[topk_local_indices]  # [block_N, max_k, C]
                        
                        # Apply valid mask: only aggregate values above threshold
                        # topk_values: [block_N, max_k], valid_topk_mask: [block_N, max_k]
                        topk_values_masked = torch.where(
                            valid_topk_mask,
                            topk_values,
                            torch.tensor(0.0, device=device, dtype=topk_values.dtype)
                        )  # [block_N, max_k]
                        
                        # Aggregate alpha: Sum(weight * evidence) for valid values only
                        # topk_values_masked: [block_N, max_k] -> [block_N, max_k, 1]
                        if aggregation_mode == 'class_conditional':
                            # CCK-Norm: Each Gaussian only contributes to its argmax class,
                            # normalized by the number of Gaussians predicting that class.
                            hard_class = gathered_evidence.argmax(dim=-1)  # [block_N, max_k]
                            max_conf = gathered_evidence.max(dim=-1)[0]    # [block_N, max_k]
                            C_dim = gathered_evidence.shape[-1]
                            block_alpha_new = torch.zeros(block_N, C_dim, device=device, dtype=gathered_evidence.dtype)
                            for c in range(C_dim):
                                c_mask = (hard_class == c) & (topk_values_masked > 0)  # [block_N, max_k]
                                c_kernel = topk_values_masked * c_mask.float()
                                numerator = (c_kernel * max_conf).sum(dim=1)  # [block_N]
                                count = c_mask.float().sum(dim=1).clamp(min=float(cc_norm_min_count))
                                block_alpha_new[:, c] = numerator / count
                        elif aggregation_mode == 'class_filtered' and base_probs is not None:
                            # Class-Filtered Kernel: For each voxel, only aggregate Gaussians
                            # whose predicted class matches the base model's prediction.
                            # This prevents majority-class Gaussians from polluting minority-class voxels.
                            base_block = base_probs[b, block_start:block_end, :]  # [block_N, C]
                            voxel_class = base_block.argmax(dim=-1)  # [block_N]
                            gauss_class = gathered_evidence.argmax(dim=-1)  # [block_N, max_k]
                            # Filter: only keep Gaussians matching voxel's class
                            class_match = (gauss_class == voxel_class.unsqueeze(1))  # [block_N, max_k]
                            filtered_kernel = topk_values_masked * class_match.float()
                            # Fallback: if no matching Gaussians, use all (standard mode)
                            no_match = (filtered_kernel.sum(dim=1) == 0)
                            if no_match.any():
                                filtered_kernel[no_match] = topk_values_masked[no_match]
                            block_alpha_new = (filtered_kernel.unsqueeze(-1) * gathered_evidence).sum(dim=1)
                        elif aggregation_mode == 'attention' and base_probs is not None:
                            # SSA: Semantic-similarity attention using base model predictions
                            base_block = base_probs[b, block_start:block_end, :]  # [block_N, C]
                            # Cosine similarity between base prediction and Gaussian evidence
                            base_norm = base_block.unsqueeze(1)  # [block_N, 1, C]
                            gauss_norm = gathered_evidence       # [block_N, max_k, C]
                            cos_sim = torch.nn.functional.cosine_similarity(
                                base_norm, gauss_norm, dim=-1)   # [block_N, max_k]
                            attn = torch.exp(cos_sim / max(attention_tau, 0.01))  # [block_N, max_k]
                            attn = attn * (topk_values_masked > 0).float()  # zero out invalid
                            weighted_kernel = topk_values_masked * attn
                            block_alpha_new = (weighted_kernel.unsqueeze(-1) * gathered_evidence).sum(dim=1)
                        elif aggregation_mode == 'cc_attention' and base_probs is not None:
                            # Combined: CCK-Norm + attention
                            hard_class = gathered_evidence.argmax(dim=-1)
                            max_conf = gathered_evidence.max(dim=-1)[0]
                            base_block = base_probs[b, block_start:block_end, :]
                            cos_sim = torch.nn.functional.cosine_similarity(
                                base_block.unsqueeze(1), gathered_evidence, dim=-1)
                            attn = torch.exp(cos_sim / max(attention_tau, 0.01))
                            attn = attn * (topk_values_masked > 0).float()
                            C_dim = gathered_evidence.shape[-1]
                            block_alpha_new = torch.zeros(block_N, C_dim, device=device, dtype=gathered_evidence.dtype)
                            for c in range(C_dim):
                                c_mask = (hard_class == c) & (topk_values_masked > 0)
                                c_kernel = topk_values_masked * c_mask.float() * attn
                                numerator = (c_kernel * max_conf).sum(dim=1)
                                count = c_mask.float().sum(dim=1).clamp(min=float(cc_norm_min_count))
                                block_alpha_new[:, c] = numerator / count
                        else:
                            # Standard aggregation
                            block_alpha_new = (topk_values_masked.unsqueeze(-1) * gathered_evidence).sum(dim=1)  # [block_N, C]
                        
                        # Aggregate uncertainty: Sum(weight * uncertainty) for valid values only
                        local_unc = u_batch[valid_g_idx]  # [K]
                        gathered_unc = local_unc[topk_local_indices]  # [block_N, max_k]
                        block_unc_new = (topk_values_masked * gathered_unc).sum(dim=1)  # [block_N]
                        
                        # Accumulate to final result
                        alpha_accum[b, block_start:block_end] += block_alpha_new
                        uncertainty_accum[b, block_start:block_end] += block_unc_new
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        perf_times['sparsify_data_extraction'] += time.time() - t_sub
                        
                        # Debug: capture selected Gaussians and kernel values for specified points
                        debug_indices = getattr(self, '_debug_point_indices', None)
                        if debug_indices is not None:
                            for n_offset in range(block_N):
                                n = block_start + n_offset
                                if n in debug_indices:
                                    valid = valid_topk_mask[n_offset, :]
                                    self._debug_capture[n] = {
                                        'gaussian_indices': topk_global_indices[n_offset, valid].cpu().tolist(),
                                        'kernel_values': topk_values_masked[n_offset, valid].cpu().tolist(),
                                    }
                        
                        del topk_values, topk_local_indices, topk_global_indices, gathered_evidence, gathered_unc, local_evidence, local_unc, block_alpha_new, block_unc_new, block_kernel_masked, topk_values_masked, valid_topk_mask
                    else:
                        # No max_kernels_per_point: use all values above threshold
                        t_sub = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        # Extract local evidence: [K, C]
                        local_evidence = w_batch[valid_g_idx]  # [K, C]
                        
                        # Aggregate alpha based on aggregation_mode
                        if aggregation_mode == 'class_conditional':
                            hard_class = local_evidence.argmax(dim=-1)  # [K]
                            max_conf = local_evidence.max(dim=-1)[0]    # [K]
                            C_dim = local_evidence.shape[-1]
                            block_alpha_new = torch.zeros(block_N, C_dim, device=device, dtype=local_evidence.dtype)
                            for c in range(C_dim):
                                c_mask = (hard_class == c).float()  # [K]
                                c_kernel = block_kernel_values_thresholded * c_mask.unsqueeze(0)  # [block_N, K]
                                numerator = (c_kernel * max_conf.unsqueeze(0)).sum(dim=1)  # [block_N]
                                count = (c_kernel > 0).float().sum(dim=1).clamp(min=float(cc_norm_min_count))
                                block_alpha_new[:, c] = numerator / count
                        elif aggregation_mode == 'class_filtered' and base_probs is not None:
                            base_block = base_probs[b, block_start:block_end, :]
                            voxel_class = base_block.argmax(dim=-1)  # [block_N]
                            gauss_class = local_evidence.argmax(dim=-1)  # [K]
                            class_match = (gauss_class.unsqueeze(0) == voxel_class.unsqueeze(1)).float()
                            filtered_kernel = block_kernel_values_thresholded * class_match
                            no_match = (filtered_kernel.sum(dim=1) == 0)
                            if no_match.any():
                                filtered_kernel[no_match] = block_kernel_values_thresholded[no_match]
                            block_alpha_new = torch.matmul(filtered_kernel, local_evidence)
                        elif aggregation_mode == 'attention' and base_probs is not None:
                            base_block = base_probs[b, block_start:block_end, :]  # [block_N, C]
                            cos_sim = torch.nn.functional.cosine_similarity(
                                base_block.unsqueeze(1),  # [block_N, 1, C]
                                local_evidence.unsqueeze(0),  # [1, K, C]
                                dim=-1)  # [block_N, K]
                            attn = torch.exp(cos_sim / max(attention_tau, 0.01))
                            weighted_kernel = block_kernel_values_thresholded * attn
                            block_alpha_new = torch.matmul(weighted_kernel, local_evidence)
                        elif aggregation_mode == 'cc_attention' and base_probs is not None:
                            hard_class = local_evidence.argmax(dim=-1)
                            max_conf = local_evidence.max(dim=-1)[0]
                            base_block = base_probs[b, block_start:block_end, :]
                            cos_sim = torch.nn.functional.cosine_similarity(
                                base_block.unsqueeze(1), local_evidence.unsqueeze(0), dim=-1)
                            attn = torch.exp(cos_sim / max(attention_tau, 0.01))
                            C_dim = local_evidence.shape[-1]
                            block_alpha_new = torch.zeros(block_N, C_dim, device=device, dtype=local_evidence.dtype)
                            for c in range(C_dim):
                                c_mask = (hard_class == c).float()
                                c_kernel = block_kernel_values_thresholded * c_mask.unsqueeze(0) * attn
                                numerator = (c_kernel * max_conf.unsqueeze(0)).sum(dim=1)
                                count = (c_kernel > 0).float().sum(dim=1).clamp(min=float(cc_norm_min_count))
                                block_alpha_new[:, c] = numerator / count
                        else:
                            block_alpha_new = torch.matmul(block_kernel_values_thresholded, local_evidence)  # [block_N, C]
                        
                        # Aggregate uncertainty: block_kernel_values_thresholded [block_N, K] @ local_unc [K] -> [block_N]
                        local_unc = u_batch[valid_g_idx]  # [K]
                        block_unc_new = torch.matmul(block_kernel_values_thresholded, local_unc)  # [block_N]
                        
                        # Accumulate to final result
                        alpha_accum[b, block_start:block_end] += block_alpha_new
                        uncertainty_accum[b, block_start:block_end] += block_unc_new
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        perf_times['sparsify_data_extraction'] += time.time() - t_sub
                        
                        del local_evidence, local_unc, block_alpha_new, block_unc_new
                    
                    del block_kernel_values_thresholded
                    
                elif sparsify_on_the_fly:
                    # Vectorized sparsification: process entire block at once
                    # block_kernel_values: [block_N, K]
                    
                    # Step 1: Apply threshold filter (vectorized for all points)
                    t_sub = time.time()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    threshold_mask = block_kernel_values > sparsify_threshold  # [block_N, K]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    perf_times['sparsify_threshold_mask'] += time.time() - t_sub
                    
                    if sparsify_max_kernels_per_point is not None:
                        # Use vectorized topk for all points at once
                        t_sub = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        # Replace values below threshold with -inf so they won't be selected
                        block_kernel_masked = torch.where(
                            threshold_mask,
                            block_kernel_values,
                            torch.tensor(float('-inf'), device=device, dtype=block_kernel_values.dtype)
                        )  # [block_N, K]
                        
                        # Get top-k for all points in parallel: [block_N, max_k]
                        max_k = min(sparsify_max_kernels_per_point, K)
                        topk_values, topk_local_indices = torch.topk(
                            block_kernel_masked,
                            k=max_k,
                            dim=1,  # Top-k along K dimension for each point
                            largest=True
                        )  # [block_N, max_k]
                        
                        # Filter out values below threshold
                        valid_topk_mask = topk_values > sparsify_threshold  # [block_N, max_k]
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        perf_times['sparsify_topk'] += time.time() - t_sub
                        
                        # Map local indices to global Gaussian indices
                        t_sub = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        # valid_g_idx: [K] -> expand to [block_N, max_k]
                        valid_g_idx_expanded = valid_g_idx.unsqueeze(0).expand(block_N, -1)  # [block_N, K]
                        # Gather global indices using topk_local_indices
                        topk_global_indices = torch.gather(
                            valid_g_idx_expanded, 
                            dim=1, 
                            index=topk_local_indices
                        )  # [block_N, max_k]
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        perf_times['sparsify_index_mapping'] += time.time() - t_sub
                        
                        # Process each point to extract valid values (still need loop for variable-length lists)
                        t_sub = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        for n_offset in range(block_N):
                            n = block_start + n_offset
                            point_valid_mask = valid_topk_mask[n_offset, :]  # [max_k]
                            
                            if point_valid_mask.any():
                                point_kernel_values = topk_values[n_offset, point_valid_mask]  # [K_n]
                                point_gaussian_indices = topk_global_indices[n_offset, point_valid_mask]  # [K_n]
                                
                                t_append = time.time()
                                kernel_values_list[b].append(point_kernel_values)
                                gaussian_indices_list[b].append(point_gaussian_indices)
                                perf_times['sparsify_list_append'] += time.time() - t_append
                                
                                num_nonzero[b, n] = len(point_kernel_values)
                            else:
                                t_append = time.time()
                                kernel_values_list[b].append(torch.tensor([], device=device, dtype=block_kernel_values.dtype))
                                gaussian_indices_list[b].append(torch.tensor([], device=device, dtype=torch.long))
                                perf_times['sparsify_list_append'] += time.time() - t_append
                                num_nonzero[b, n] = 0
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        perf_times['sparsify_data_extraction'] += time.time() - t_sub
                        
                        del block_kernel_masked, topk_values, topk_local_indices, valid_topk_mask, topk_global_indices, valid_g_idx_expanded
                    else:
                        # No max_kernels_per_point: extract all values above threshold
                        t_sub = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        for n_offset in range(block_N):
                            n = block_start + n_offset
                            point_mask = threshold_mask[n_offset, :]  # [K]
                            
                            if point_mask.any():
                                point_kernel_values = block_kernel_values[n_offset, point_mask]  # [K_n]
                                point_gaussian_indices = valid_g_idx[point_mask]  # [K_n]
                                
                                t_append = time.time()
                                kernel_values_list[b].append(point_kernel_values)
                                gaussian_indices_list[b].append(point_gaussian_indices)
                                perf_times['sparsify_list_append'] += time.time() - t_append
                                
                                num_nonzero[b, n] = len(point_kernel_values)
                            else:
                                t_append = time.time()
                                kernel_values_list[b].append(torch.tensor([], device=device, dtype=block_kernel_values.dtype))
                                gaussian_indices_list[b].append(torch.tensor([], device=device, dtype=torch.long))
                                perf_times['sparsify_list_append'] += time.time() - t_append
                                num_nonzero[b, n] = 0
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        perf_times['sparsify_data_extraction'] += time.time() - t_sub
                    
                    del threshold_mask
                else:
                    # For dense format, map back to full [block_N, G] tensor
                    block_kernel_dense = torch.zeros(block_N, G, device=device, dtype=query_points.dtype)
                    block_kernel_dense[:, valid_g_idx] = block_kernel_values  # [block_N, G]
                    kernel_values[b, block_start:block_end, :] = block_kernel_dense
                    del block_kernel_dense
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                perf_times['sparsify'] += time.time() - t_start
                
                t_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                del block_kernel_values
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                perf_times['memory_ops'] += time.time() - t_start
        
        # Cleanup
        t_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del cov_inv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        perf_times['memory_ops'] += time.time() - t_start
        
        # Print performance breakdown only in verbose mode.
        total_time = sum(perf_times.values())
        if self.verbose and total_time > 0:
            print("\n" + "="*80)
            print("Blockwise Pruning Performance Breakdown")
            print("="*80)
            print(f"{'Operation':<35} {'Time (s)':<15} {'Percentage':<15} {'Count/Info':<20}")
            print("-"*80)
            
            # Separate sparsify sub-operations
            sparsify_sub_ops = ['sparsify_threshold_mask', 'sparsify_topk', 'sparsify_index_mapping', 
                              'sparsify_data_extraction', 'sparsify_list_append']
            sparsify_total = sum(perf_times[k] for k in sparsify_sub_ops)
            
            for key, time_val in sorted(perf_times.items(), key=lambda x: x[1], reverse=True):
                if key in sparsify_sub_ops:
                    continue  # Skip sub-ops, will print separately
                percentage = (time_val / total_time) * 100 if total_time > 0 else 0
                count_info = ""
                if key == 'bounding_box':
                    count_info = f"{perf_counts['num_blocks']} blocks"
                elif key == 'spatial_pruning':
                    count_info = f"{perf_counts['num_blocks']} blocks"
                elif key == 'mahalanobis_dist':
                    count_info = f"{perf_counts['total_valid_gaussians']} ops"
                elif key == 'sparsify':
                    count_info = f"{N} points (total)"
                print(f"{key:<35} {time_val:<15.4f} {percentage:<15.2f}% {count_info:<20}")
            
            # Print sparsify sub-operations
            if sparsify_total > 0:
                print("-"*80)
                print(f"{'  -> sparsify (total)':<35} {sparsify_total:<15.4f} {(sparsify_total/total_time*100):<15.2f}%")
                for key in sparsify_sub_ops:
                    if perf_times[key] > 0:
                        time_val = perf_times[key]
                        percentage = (time_val / sparsify_total) * 100 if sparsify_total > 0 else 0
                        print(f"    - {key:<31} {time_val:<15.4f} {percentage:<15.2f}% (of sparsify)")
            
            print("-"*80)
            print(f"{'TOTAL':<35} {total_time:<15.4f} {'100.00':<15}%")
            print(f"\nStatistics:")
            print(f"  Total blocks: {perf_counts['num_blocks']}")
            print(f"  Empty blocks: {perf_counts['num_empty_blocks']}")
            print(f"  Valid Gaussians processed: {perf_counts['total_valid_gaussians']}")
            print(f"  Filtered Gaussians: {perf_counts['total_filtered_gaussians']}")
            total = B * N * G
            reduction = (1 - perf_counts['total_valid_gaussians'] / total) * 100 if total > 0 else 0
            print(f"  Computation reduction: {reduction:.1f}%")
            if direct_aggregation:
                print(f"  Mode: Direct aggregation (alpha + uncertainty)")
            print("="*80 + "\n")
        
        if self.verbose:
            total = B * N * G
            percentage = (total_filtered / total) * 100 if total > 0 else 0
            reduction = (1 - total_gaussians_processed / total) * 100 if total > 0 else 0
            print(f"[AnisotropicKernel] Blockwise pruning: {total_filtered}/{total} "
                  f"({percentage:.2f}%) Gaussians within distance, "
                  f"computation reduced by {reduction:.1f}%")
            if direct_aggregation:
                print(f"[AnisotropicKernel] Using direct aggregation mode (no Python lists)")
            elif sparsify_on_the_fly:
                mean_nonzero = num_nonzero.float().mean().item()
                max_nonzero = num_nonzero.max().item()
                print(f"[AnisotropicKernel] On-the-fly sparsification: mean_nonzero={mean_nonzero:.1f}, max_nonzero={max_nonzero}")
        
        # Return format based on mode
        if direct_aggregation:
            if self.timing_profiler is not None:
                self.timing_profiler.end('compute_kernel_batch')
            return {
                'alpha_increment': alpha_accum,  # [B, N, C]
                'uncertainty_increment': uncertainty_accum,  # [B, N]
                'direct_aggregation': True
            }
        elif sparsify_on_the_fly:
            if self.timing_profiler is not None:
                self.timing_profiler.end('compute_kernel_batch')
            return {
                'kernel_values': kernel_values_list,
                'gaussian_indices': gaussian_indices_list,
                'num_nonzero': num_nonzero,
                'threshold': sparsify_threshold,
                'max_kernels_per_point': sparsify_max_kernels_per_point,
                'device': device,
                'dtype': query_points.dtype
            }
        else:
            if self.timing_profiler is not None:
                self.timing_profiler.end('compute_kernel_batch')
            return kernel_values
    
    def k_prime(self, d, ell):
        """
        Paper's base kernel function k'(d, ℓ) from Equation (2).
        
        Formula (2):
            k'(d, ℓ) = {
                1/3 * (1 - d/ℓ) * [2 + cos(2πd/ℓ)] + 1/(2π) * sin(2πd/ℓ)  if d < ℓ
                0                                                              otherwise
            }
        
        Args:
            d: Distance (tensor) - can be scalar, vector, or any shape
            ell: Kernel scale ℓ (tensor) - should be broadcastable with d
        
        Returns:
            kernel_value: Same shape as d
        """
        # Ensure d and ell are tensors with compatible shapes
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float32)
        if not isinstance(ell, torch.Tensor):
            ell = torch.tensor(ell, dtype=d.dtype, device=d.device)
        
        # Broadcast ell to match d's shape
        ell = ell.expand_as(d) if ell.shape != d.shape else ell
        
        # Compute indicator: 1 if d < ℓ, 0 otherwise
        indicator = (d < ell).float()
        
        # Compute normalized distance
        d_normalized = d / (ell + self.eps)  # d/ℓ
        
        # Compute kernel value: k'(d, ℓ) = 1/3 * (1 - d/ℓ) * [2 + cos(2πd/ℓ)] + 1/(2π) * sin(2πd/ℓ)
        term1 = (1 - d_normalized) / 3.0  # (1 - d/ℓ) / 3
        term2 = 2 + torch.cos(2 * torch.pi * d_normalized)  # 2 + cos(2πd/ℓ)
        term3 = torch.sin(2 * torch.pi * d_normalized) / (2 * torch.pi)  # sin(2πd/ℓ) / (2π)
        
        kernel_value = term1 * term2 + term3
        
        # Apply indicator: kernel = 0 if d >= ℓ
        kernel_value = kernel_value * indicator
        
        return kernel_value
    
    def compute_ellipsoid_surface_distance(self, query_points, gaussian_means, gaussian_covariances, tau=1.0):
        """
        Compute distance from query points to ellipsoid surface (Equation 7 approximation).
        
        Paper Equation (7):
            min ||x̂m - v||²  s.t.  (v - µj)ᵀ Σj⁻¹ (v - µj) = τ
        
        For computational efficiency, we use an analytical approximation:
        - Compute Mahalanobis distance: d_maha² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        - If d_maha² < τ: point is inside ellipsoid, distance = 0
        - If d_maha² >= τ: approximate distance to surface
        
        Args:
            query_points: [N, 3] or [B, N, 3] - Query points
            gaussian_means: [G, 3] or [B, G, 3] - Gaussian centers
            gaussian_covariances: [G, 3, 3] or [B, G, 3, 3] - Covariance matrices
            tau: Ellipsoid size parameter τ (default: 1.0)
        
        Returns:
            distances: [B, N, G] - Distances to ellipsoid surface (batched)
        """
        # Input validation (must be batched)
        if len(query_points.shape) != 3 or query_points.shape[2] != 3:
            raise ValueError(f"query_points must be [B, N, 3], got {query_points.shape}")
        if len(gaussian_means.shape) != 3 or gaussian_means.shape[2] != 3:
            raise ValueError(f"gaussian_means must be [B, G, 3], got {gaussian_means.shape}")
        if len(gaussian_covariances.shape) != 4 or gaussian_covariances.shape[2:] != (3, 3):
            raise ValueError(f"gaussian_covariances must be [B, G, 3, 3], got {gaussian_covariances.shape}")
        
        B, N, _ = query_points.shape
        _, G, _, _ = gaussian_covariances.shape
        device = query_points.device
        
        # Compute difference: diff = x - μ
        # [B, N, 1, 3] - [B, 1, G, 3] = [B, N, G, 3]
        query_expanded = query_points.unsqueeze(2)  # [B, N, 1, 3]
        means_expanded = gaussian_means.unsqueeze(1)  # [B, 1, G, 3]
        diff = query_expanded - means_expanded  # [B, N, G, 3]
        
        # Add regularization to covariance matrices
        eye = torch.eye(3, dtype=gaussian_covariances.dtype, device=device)
        cov_reg = gaussian_covariances + self.eps * eye.view(1, 1, 3, 3)  # [B, G, 3, 3]
        
        # Expand for batch multiplication: [B, 1, G, 3, 3]
        cov_reg_expanded = cov_reg.unsqueeze(1)  # [B, 1, G, 3, 3]
        
        # Compute inverse
        try:
            cov_inv = torch.inverse(cov_reg_expanded)  # [B, 1, G, 3, 3]
        except RuntimeError:
            cov_inv = torch.pinverse(cov_reg_expanded)
        
        # Compute squared Mahalanobis distance: d² = diff^T * Cov^(-1) * diff
        diff_expanded = diff.unsqueeze(-1)  # [B, N, G, 3, 1]
        cov_inv_diff = torch.matmul(cov_inv, diff_expanded)  # [B, N, G, 3, 1]
        mahalanobis_dist_sq = torch.sum(diff * cov_inv_diff.squeeze(-1), dim=-1)  # [B, N, G]
        
        # Clamp to avoid numerical issues
        mahalanobis_dist_sq = torch.clamp(mahalanobis_dist_sq, min=0.0, max=self.max_mahalanobis_dist ** 2)
        mahalanobis_dist = torch.sqrt(mahalanobis_dist_sq + self.eps)  # [B, N, G]
        
        # For points inside ellipsoid (d_maha² < τ): distance = 0
        # For points outside: approximate distance to surface
        # We use a simple approximation: distance ≈ (d_maha - sqrt(τ)) * scale_factor
        # where scale_factor is approximated from the ellipsoid's principal axes
        tau_sqrt = torch.sqrt(torch.tensor(tau, dtype=mahalanobis_dist.dtype, device=device))
        
        # Approximate distance to ellipsoid surface
        # If inside ellipsoid (d_maha < sqrt(τ)): d = 0
        # If outside: d ≈ (d_maha - sqrt(τ)) * average_scale
        inside_mask = mahalanobis_dist < tau_sqrt
        distances = torch.zeros_like(mahalanobis_dist)
        

        #TODO:这里开始是GPT给出的近似距离计算方法，需要测试确认
        euclid = torch.linalg.norm(diff, dim=-1)  # [B, N, G]
        # mahalanobis_dist = sqrt(delta^T Sigma^{-1} delta) 你已有
        s = tau_sqrt / (mahalanobis_dist + self.eps)         # [B, N, G]
        outside_distances = (1.0 - s) * euclid               # [B, N, G]
        outside_distances = torch.clamp(outside_distances, min=0.0)

        distances = torch.where(inside_mask, torch.zeros_like(outside_distances), outside_distances)
        distances = torch.clamp(distances, min=0.0)
        
        # # For outside points, compute approximate distance
        # # We use the average scale of the covariance matrix as a scaling factor
        # # Trace of covariance matrix gives average variance
        # # cov_reg is [B, 1, G, 3, 3], need to extract [B, G, 3, 3]
        # cov_reg_for_trace = cov_reg.squeeze(1) if cov_reg.dim() == 5 else cov_reg  # [B, G, 3, 3]
        # traces = torch.diagonal(cov_reg_for_trace, dim1=-2, dim2=-1).sum(dim=-1)  # [B, G]
        # avg_scales = torch.sqrt(traces / 3.0 + self.eps)  # [B, G]
        # avg_scales = avg_scales.unsqueeze(1)  # [B, 1, G]
        
        # # Approximate distance: (d_maha - sqrt(τ)) * avg_scale
        # outside_distances = (mahalanobis_dist - tau_sqrt) * avg_scales
        # distances = torch.where(inside_mask, torch.zeros_like(distances), outside_distances)
        
        return distances
    
    def compute_covariance_matrix(self, scales, rotations):
        """
        Compute Gaussian covariance matrices.
        
        Formula: Cov = R * S * S^T * R^T
        where S = diag(scale_x, scale_y, scale_z) and R is rotation matrix.
        
        Args:
            scales: [B, G, 3] - Gaussian scales (batched)
            rotations: [B, G, 4] - Rotation quaternions [w, x, y, z] (batched)
        
        Returns:
            covariances: [B, G, 3, 3] - Covariance matrices (batched)
        """
        if self.timing_profiler is not None:
            self.timing_profiler.start('compute_covariance_matrix')
        # Input validation
        if scales is None or rotations is None:
            raise ValueError("scales and rotations cannot be None")
        
        if len(scales.shape) != 3 or scales.shape[2] != 3:
            raise ValueError(f"scales must be [B, G, 3], got {scales.shape}")
        
        if len(rotations.shape) != 3 or rotations.shape[2] != 4:
            raise ValueError(f"rotations must be [B, G, 4], got {rotations.shape}")
        
        if scales.shape[0] != rotations.shape[0]:
            raise ValueError(f"Batch size mismatch: scales batch={scales.shape[0]}, "
                           f"rotations batch={rotations.shape[0]}")
        
        if scales.shape[1] != rotations.shape[1]:
            raise ValueError(f"Number of Gaussians mismatch: scales G={scales.shape[1]}, "
                           f"rotations G={rotations.shape[1]}")
        
        # Check for NaN and Inf
        if torch.isnan(scales).any():
            nan_indices = torch.nonzero(torch.isnan(scales))
            raise ValueError(f"scales contains NaN at indices: {nan_indices.tolist()}")
        
        if torch.isinf(scales).any():
            inf_indices = torch.nonzero(torch.isinf(scales))
            raise ValueError(f"scales contains Inf at indices: {inf_indices.tolist()}")
        
        if torch.isnan(rotations).any():
            nan_indices = torch.nonzero(torch.isnan(rotations))
            raise ValueError(f"rotations contains NaN at indices: {nan_indices.tolist()}")
        
        if torch.isinf(rotations).any():
            inf_indices = torch.nonzero(torch.isinf(rotations))
            raise ValueError(f"rotations contains Inf at indices: {inf_indices.tolist()}")
        
        # Check scale values (should be positive)
        if (scales <= 0).any():
            invalid_indices = torch.nonzero(scales <= 0)
            invalid_values = scales[scales <= 0]
            raise ValueError(f"scales must be positive, found non-positive values at indices: "
                           f"{invalid_indices.tolist()}, values: {invalid_values.tolist()}")
        
        # Check scale range (warn if too small or too large)
        if self.verbose:
            min_scale = scales.min().item()
            max_scale = scales.max().item()
            if min_scale < 1e-6:
                print(f"[AnisotropicKernel] Warning: Very small scale detected: min={min_scale:.2e}")
            if max_scale > 100.0:
                print(f"[AnisotropicKernel] Warning: Very large scale detected: max={max_scale:.2f}")
        
        # Check quaternion normalization
        quaternion_norms = torch.norm(rotations, dim=-1)
        if not torch.allclose(quaternion_norms, torch.ones_like(quaternion_norms), atol=1e-3):
            invalid_norms = quaternion_norms[torch.abs(quaternion_norms - 1.0) > 1e-3]
            invalid_indices = torch.nonzero(torch.abs(quaternion_norms - 1.0) > 1e-3)
            if self.verbose:
                print(f"[AnisotropicKernel] Warning: Quaternions not normalized. "
                      f"Found {len(invalid_indices)} quaternions with norms: {invalid_norms.tolist()}")
                print(f"  Invalid indices: {invalid_indices.tolist()}")
                print(f"  Auto-normalizing quaternions...")
            # Auto-normalize
            rotations = rotations / (quaternion_norms.unsqueeze(-1) + self.eps)
        
        B, G, _ = scales.shape
        
        # Step 1: Build scale diagonal matrix S
        # S = diag(scale_x, scale_y, scale_z)
        S = torch.zeros(B, G, 3, 3, dtype=scales.dtype, device=scales.device)
        S[..., 0, 0] = scales[..., 0]  # scale_x
        S[..., 1, 1] = scales[..., 1]  # scale_y
        S[..., 2, 2] = scales[..., 2]  # scale_z
        
        # Step 2: Convert quaternions to rotation matrices R
        # rotations: [B, G, 4] -> R: [B, G, 3, 3]
        R = get_rotation_matrix(rotations)  # [B, G, 3, 3]
        
        # Step 3: Compute S * S^T (since S is diagonal, S * S^T = S^2)
        # For diagonal matrix: S * S^T = diag(scale_x^2, scale_y^2, scale_z^2)
        S_squared = S * S  # Element-wise square (equivalent to S * S^T for diagonal)
        
        # Step 4: Compute Cov = R * S * S^T * R^T = R * S_squared * R^T
        # First: R * S_squared
        R_S_squared = torch.matmul(R, S_squared)  # [B, G, 3, 3]
        # Then: (R * S_squared) * R^T
        Cov = torch.matmul(R_S_squared, R.transpose(-1, -2))  # [B, G, 3, 3]
        
        # Validate covariance matrices
        if torch.isnan(Cov).any():
            nan_indices = torch.nonzero(torch.isnan(Cov))
            raise ValueError(f"Covariance matrix contains NaN at indices: {nan_indices.tolist()}")
        
        if torch.isinf(Cov).any():
            inf_indices = torch.nonzero(torch.isinf(Cov))
            raise ValueError(f"Covariance matrix contains Inf at indices: {inf_indices.tolist()}")
        
        # Check symmetry (covariance matrices should be symmetric)
        if self.verbose:
            # Check first few covariance matrices
            num_check = min(5, Cov.shape[0])
            for idx in range(num_check):
                cov_i = Cov[idx]
                if not torch.allclose(cov_i, cov_i.transpose(-1, -2), atol=1e-4):
                    print(f"[AnisotropicKernel] Warning: Covariance matrix at index {idx} is not symmetric")
        
        if self.timing_profiler is not None:
            self.timing_profiler.end('compute_covariance_matrix')
        return Cov
    
    def compute_kernel_batch(self, query_points, gaussian_means, gaussian_covariances,
                             ell=None, use_ellipsoid_distance=False, tau=1.0,
                             sparsify_on_the_fly=False, sparsify_threshold=1e-6, sparsify_max_kernels_per_point=None,
                             uncertainty_mask=None, semantics=None, gaussian_uncertainties=None,
                             aggregation_mode='standard', cc_norm_min_count=1,
                             base_probs=None, attention_tau=0.3):
        """
        Batch version of kernel computation.
        """
        if self.timing_profiler is not None:
            self.timing_profiler.start('compute_kernel_batch')
        """
        
        If sparsify_on_the_fly=True and streaming chunking is enabled, directly outputs sparse format
        without creating full dense tensor, saving significant memory (e.g., 31.25GB -> ~0.04GB).
        
        If uncertainty_mask is provided, it will be applied at chunk level during streaming chunking,
        allowing on-the-fly sparsification even with uncertainty-adaptive kernels.
        
        Args:
            query_points: [B, N, 3] - Batch of query points
            gaussian_means: [B, G, 3] - Batch of Gaussian centers
            gaussian_covariances: [B, G, 3, 3] - Batch of covariance matrices
            ell: Kernel scale ℓ. If None, uses self.scale. Can be [B, G] for per-batch-per-Gaussian scales.
            use_ellipsoid_distance: Whether to use ellipsoid surface distance (Equation 7) instead of Mahalanobis distance
            tau: Ellipsoid size parameter τ for ellipsoid distance (default: 1.0)
            sparsify_on_the_fly: If True and streaming chunking is used, directly output sparse format
                to avoid creating full dense tensor (default: False). This can save ~99.9% memory.
            sparsify_threshold: Threshold for on-the-fly sparsification (only used if sparsify_on_the_fly=True)
            sparsify_max_kernels_per_point: Max kernels per point for on-the-fly sparsification
                (only used if sparsify_on_the_fly=True)
            uncertainty_mask: [B, G] or None - Optional uncertainty mask to apply at chunk level.
                If provided and sparsify_on_the_fly=True, mask is applied before sparsification.
                This allows combining uncertainty filtering with on-the-fly sparsification.
        
        Returns:
            kernel_values: [B, N, G] (dense) or dict (sparse) - Kernel values
                If sparsify_on_the_fly=True and streaming is used, returns sparse format dict
        """
        # Input validation
        if query_points is None or gaussian_means is None or gaussian_covariances is None:
            raise ValueError("query_points, gaussian_means, and gaussian_covariances cannot be None")
        
        if len(query_points.shape) != 3 or query_points.shape[2] != 3:
            raise ValueError(f"query_points shape must be [B, N, 3], got {query_points.shape}")
        
        if len(gaussian_means.shape) != 3 or gaussian_means.shape[2] != 3:
            raise ValueError(f"gaussian_means shape must be [B, G, 3], got {gaussian_means.shape}")
        
        if len(gaussian_covariances.shape) != 4 or gaussian_covariances.shape[2:] != (3, 3):
            raise ValueError(f"gaussian_covariances shape must be [B, G, 3, 3], got {gaussian_covariances.shape}")
        
        if query_points.shape[0] != gaussian_means.shape[0]:
            raise ValueError(f"Batch size mismatch: query_points B={query_points.shape[0]}, "
                           f"gaussian_means B={gaussian_means.shape[0]}")
        
        if gaussian_means.shape[0] != gaussian_covariances.shape[0]:
            raise ValueError(f"Batch size mismatch: gaussian_means B={gaussian_means.shape[0]}, "
                           f"gaussian_covariances B={gaussian_covariances.shape[0]}")
        
        if gaussian_means.shape[1] != gaussian_covariances.shape[1]:
            raise ValueError(f"Number of Gaussians mismatch: means G={gaussian_means.shape[1]}, "
                           f"covariances G={gaussian_covariances.shape[1]}")
        
        # Check device consistency
        devices = [query_points.device, gaussian_means.device, gaussian_covariances.device]
        if len(set(str(d) for d in devices)) > 1:
            raise ValueError(f"Device mismatch: query_points={query_points.device}, "
                           f"gaussian_means={gaussian_means.device}, "
                           f"gaussian_covariances={gaussian_covariances.device}")
        
        # Check for NaN and Inf
        if torch.isnan(query_points).any():
            nan_indices = torch.nonzero(torch.isnan(query_points))
            raise ValueError(f"query_points contains NaN at batch/point indices: {nan_indices.tolist()[:10]}")
        
        if torch.isnan(gaussian_means).any():
            nan_indices = torch.nonzero(torch.isnan(gaussian_means))
            raise ValueError(f"gaussian_means contains NaN at batch/gaussian indices: {nan_indices.tolist()[:10]}")
        
        if torch.isnan(gaussian_covariances).any():
            nan_indices = torch.nonzero(torch.isnan(gaussian_covariances))
            raise ValueError(f"gaussian_covariances contains NaN at indices: {nan_indices.tolist()[:10]}")
        
        B, N, _ = query_points.shape
        _, G, _, _ = gaussian_covariances.shape
        device = query_points.device
        
        # Determine kernel scale ℓ
        if ell is None:
            ell = self.scale
        
        # Convert ell to tensor if needed
        if not isinstance(ell, torch.Tensor):
            ell = torch.tensor(ell, dtype=query_points.dtype, device=device)
        
        # Ensure ell is broadcastable: [B, G] or [G] or scalar
        if ell.dim() == 0:
            ell = ell.expand(B, G)
        elif ell.dim() == 1:
            if ell.shape[0] == G:
                ell = ell.unsqueeze(0).expand(B, G)  # [G] -> [1, G] -> [B, G]
            elif ell.shape[0] == B:
                ell = ell.unsqueeze(1).expand(B, G)  # [B] -> [B, 1] -> [B, G]
            else:
                raise ValueError(f"ell shape {ell.shape} must be [G] or [B] or scalar")
        elif ell.shape != (B, G):
            raise ValueError(f"ell shape {ell.shape} must be [B, G]")
        
        # Determine whether to use blockwise spatial pruning
        # Blockwise pruning is more efficient when:
        # 1. Blockwise pruning is enabled (use_blockwise_pruning=True)
        # 2. Distance filtering is enabled (max_euclidean_distance > 0)
        # 3. N is large enough to benefit from blockwise processing
        use_blockwise = (self.use_blockwise_pruning and 
                        self.max_euclidean_distance is not None and 
                        self.max_euclidean_distance > 0 and 
                        N > self.block_size)
        
        # Debug: capture per-point selected Gaussians and kernel values for comparison
        if getattr(self, '_debug_point_indices', None) is not None:
            self._debug_capture = {}
        
        if use_blockwise:
            # Use blockwise spatial pruning: group query points into blocks,
            # compute bounding boxes, and only process Gaussians within bounding boxes
            if self.verbose:
                num_blocks = (N + self.block_size - 1) // self.block_size
                print(f"[AnisotropicKernel] Using blockwise spatial pruning: "
                      f"N={N}, block_size={self.block_size}, num_blocks={num_blocks}")
            
            return self._compute_sparse_kernel_blockwise(
                query_points, gaussian_means, gaussian_covariances,
                ell, use_ellipsoid_distance, tau,
                sparsify_on_the_fly, sparsify_threshold, sparsify_max_kernels_per_point,
                uncertainty_mask, semantics=semantics, gaussian_uncertainties=gaussian_uncertainties,
                aggregation_mode=aggregation_mode, cc_norm_min_count=cc_norm_min_count,
                base_probs=base_probs, attention_tau=attention_tau
            )
        
        # Determine whether to use streaming chunking
        # Use streaming chunking if:
        # 1. Streaming is enabled (use_streaming_chunking=True)
        # 2. Distance filtering is enabled (max_euclidean_distance > 0)
        # 3. N is large enough to benefit from chunking
        max_chunk_size = 4096  # Process at most 4096 query points at a time (~1.2GB for G=25600)
        use_streaming = (self.use_streaming_chunking and 
                        self.max_euclidean_distance is not None and 
                        self.max_euclidean_distance > 0 and 
                        N > max_chunk_size)
        
        if use_streaming:
            # Streaming chunking mode: process chunks independently, accumulate kernel values
            # Never store full distance_mask or distances tensors in memory
            if self.verbose:
                num_chunks = (N + max_chunk_size - 1) // max_chunk_size
                print(f"[AnisotropicKernel] Using streaming chunking: "
                      f"N={N}, chunk_size={max_chunk_size}, num_chunks={num_chunks}")
            
            # Pre-compute covariance inverse (reused for all chunks)
            eye = torch.eye(3, dtype=gaussian_covariances.dtype, device=device)
            cov_reg = gaussian_covariances + self.eps * eye.view(1, 1, 3, 3)  # [B, G, 3, 3]
            try:
                cov_inv = torch.inverse(cov_reg)  # [B, G, 3, 3]
            except RuntimeError:
                if self.verbose:
                    dets = torch.det(cov_reg)
                    singular_mask = torch.abs(dets) < 1e-6
                    if singular_mask.any():
                        num_singular = singular_mask.sum().item()
                        print(f"[AnisotropicKernel] Warning: {num_singular} singular covariance matrices detected")
                cov_inv = torch.pinverse(cov_reg)  # [B, G, 3, 3]
            
            # Initialize output: either dense tensor or sparse format
            means_expanded = gaussian_means.unsqueeze(1)  # [B, 1, G, 3]
            ell_expanded = ell.unsqueeze(1)  # [B, 1, G]
            total_filtered = 0
            
            if sparsify_on_the_fly:
                # Initialize sparse format: directly sparsify during chunking to avoid full dense tensor
                if self.verbose:
                    print(f"[AnisotropicKernel] Using on-the-fly sparsification: threshold={sparsify_threshold}, "
                          f"max_kernels_per_point={sparsify_max_kernels_per_point}")
                kernel_values_list = [[] for _ in range(B)]  # List[List[Tensor]] for each batch
                gaussian_indices_list = [[] for _ in range(B)]  # List[List[Tensor]] for each batch
                num_nonzero = torch.zeros(B, N, device=device, dtype=torch.long)
                if getattr(self, '_debug_point_indices', None) is not None:
                    self._debug_capture = {}
            else:
                # Initialize dense tensor
                kernel_values = torch.zeros(B, N, G, device=device, dtype=query_points.dtype)
            
            # Process each chunk independently
            for chunk_start in range(0, N, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, N)
                chunk_query = query_points[:, chunk_start:chunk_end, :]  # [B, chunk_N, 3]
                chunk_N = chunk_query.shape[1]
                
                # Step 1: Compute Euclidean distances and distance mask for this chunk
                # KEY OPTIMIZATION: Filter out Gaussians beyond max_euclidean_distance BEFORE computing expensive Mahalanobis distance
                # This significantly reduces computation for distant Gaussians
                chunk_query_expanded = chunk_query.unsqueeze(2)  # [B, chunk_N, 1, 3]
                chunk_diff = chunk_query_expanded - means_expanded  # [B, chunk_N, G, 3]
                chunk_euclidean_distances = torch.norm(chunk_diff, dim=-1)  # [B, chunk_N, G]
                chunk_mask = chunk_euclidean_distances <= self.max_euclidean_distance  # [B, chunk_N, G]
                total_filtered += chunk_mask.sum().item()
                
                # Step 2: Compute distances (Mahalanobis or ellipsoid) ONLY for Gaussians within max_euclidean_distance
                # For Gaussians beyond max_euclidean_distance, directly set distance to inf (skip expensive computation)
                if use_ellipsoid_distance:
                    # Use ellipsoid surface distance (Equation 7)
                    # Only compute for Gaussians within max_euclidean_distance
                    chunk_ellipsoid_distances = self.compute_ellipsoid_surface_distance(
                        chunk_query, gaussian_means, gaussian_covariances, tau=tau
                    )  # [B, chunk_N, G]
                    chunk_distances = torch.where(chunk_mask, chunk_ellipsoid_distances,
                                                 torch.tensor(float('inf'), device=device, dtype=query_points.dtype))
                    del chunk_ellipsoid_distances
                else:
                    # Compute Mahalanobis distance
                    # Note: Due to PyTorch vectorization, we still compute for all Gaussians,
                    # but we use the mask to filter results, avoiding kernel computation for distant Gaussians
                    cov_inv_expanded = cov_inv.unsqueeze(1)  # [B, 1, G, 3, 3]
                    chunk_diff_expanded = chunk_diff.unsqueeze(-1)  # [B, chunk_N, G, 3, 1]
                    cov_inv_diff = torch.matmul(cov_inv_expanded, chunk_diff_expanded)  # [B, chunk_N, G, 3, 1]
                    cov_inv_diff = cov_inv_diff.squeeze(-1)  # [B, chunk_N, G, 3]
                    chunk_mahalanobis_dist_sq = torch.sum(chunk_diff * cov_inv_diff, dim=-1)  # [B, chunk_N, G]
                    chunk_mahalanobis_dist_sq = torch.clamp(chunk_mahalanobis_dist_sq, min=0.0, max=self.max_mahalanobis_dist ** 2)
                    chunk_mahalanobis_dist = torch.sqrt(chunk_mahalanobis_dist_sq + self.eps)  # [B, chunk_N, G]
                    
                    # Apply distance mask: set distance to inf for Gaussians beyond max_euclidean_distance
                    # This ensures these Gaussians get kernel value = 0 (from k_prime function)
                    chunk_distances = torch.where(chunk_mask, chunk_mahalanobis_dist,
                                                 torch.tensor(float('inf'), device=device, dtype=query_points.dtype))
                    
                    # Free intermediate tensors
                    del chunk_diff, chunk_diff_expanded, cov_inv_diff, chunk_mahalanobis_dist_sq, chunk_mahalanobis_dist
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Step 3: Compute kernel values for this chunk
                # Before computing kernel, replace inf values with large finite value to avoid NaN
                # This is safe because these values will be masked out anyway
                chunk_distances_safe = torch.where(torch.isinf(chunk_distances),
                                                   torch.tensor(1e10, device=device, dtype=chunk_distances.dtype),
                                                   chunk_distances)
                
                if self.use_paper_kernel:
                    chunk_kernel_values = self.k_prime(chunk_distances_safe, ell_expanded)  # [B, chunk_N, G]
                else:
                    chunk_dist_sq = chunk_distances_safe ** 2
                    chunk_dist_sq = torch.clamp(chunk_dist_sq, min=0.0, max=self.max_mahalanobis_dist ** 2)
                    if ell.shape == (B, G):
                        chunk_kernel_values = torch.exp(-0.5 * chunk_dist_sq) * ell_expanded  # [B, chunk_N, G]
                    else:
                        chunk_kernel_values = torch.exp(-0.5 * chunk_dist_sq) * self.scale  # [B, chunk_N, G]
                    del chunk_dist_sq
                
                # Apply distance mask: set kernel values to 0 for Gaussians outside max_euclidean_distance
                # This ensures that masked Gaussians (which had inf distances) get 0 kernel values
                chunk_kernel_values = chunk_kernel_values * chunk_mask.float()  # [B, chunk_N, G]
                
                # Apply uncertainty_mask if provided (at chunk level, before sparsification)
                if uncertainty_mask is not None:
                    uncertainty_mask_expanded = uncertainty_mask.unsqueeze(1).float()  # [B, 1, G]
                    chunk_kernel_values = chunk_kernel_values * uncertainty_mask_expanded  # [B, chunk_N, G]
                
                del chunk_distances_safe
                
                # Step 4: Accumulate kernel values (either dense or sparse)
                if sparsify_on_the_fly:
                    # Vectorized on-the-fly sparsification: process entire chunk at once
                    # This is MUCH faster than looping over each query point
                    for b in range(B):
                        chunk_kernel_b = chunk_kernel_values[b, :, :]  # [chunk_N, G]
                        
                        # Vectorized threshold filtering for entire chunk
                        chunk_mask_b = chunk_kernel_b > sparsify_threshold  # [chunk_N, G]
                        
                        # Apply max_kernels_per_point if set using vectorized topk
                        if sparsify_max_kernels_per_point is not None:
                            # Use topk to get top-k kernels for each query point in parallel
                            # chunk_kernel_b: [chunk_N, G], we want top-k along G dimension
                            # Replace values below threshold with -inf so they won't be selected
                            chunk_kernel_b_masked = torch.where(
                                chunk_mask_b,
                                chunk_kernel_b,
                                torch.tensor(float('-inf'), device=device, dtype=chunk_kernel_b.dtype)
                            )
                            
                            # Get top-k kernels for each query point: [chunk_N, max_k]
                            topk_values, topk_indices = torch.topk(
                                chunk_kernel_b_masked,
                                k=min(sparsify_max_kernels_per_point, G),
                                dim=1,
                                largest=True
                            )
                            
                            # Remove -inf values (these correspond to kernels below threshold)
                            valid_mask = topk_values > sparsify_threshold  # [chunk_N, max_k]
                            
                            # Vectorized extraction: use mask to filter in-place, avoiding explicit loop where possible
                            # Note: Still need loop for appending to lists, but operations inside are vectorized
                            chunk_N_actual = chunk_end - chunk_start
                            for n_offset in range(chunk_N_actual):
                                n = chunk_start + n_offset
                                valid_row = valid_mask[n_offset, :]  # [max_k]
                                
                                # Use vectorized boolean indexing (more efficient than loop)
                                if valid_row.any():
                                    # Extract valid elements using boolean mask (vectorized)
                                    point_kernel_values = topk_values[n_offset, valid_row]  # [K_n]
                                    point_gaussian_indices = topk_indices[n_offset, valid_row]  # [K_n]
                                    
                                    kernel_values_list[b].append(point_kernel_values)
                                    gaussian_indices_list[b].append(point_gaussian_indices)
                                    num_nonzero[b, n] = len(point_kernel_values)
                                    # Debug: capture for specified points
                                    if getattr(self, '_debug_point_indices', None) is not None and n in self._debug_point_indices:
                                        self._debug_capture[n] = {
                                            'gaussian_indices': point_gaussian_indices.cpu().tolist(),
                                            'kernel_values': point_kernel_values.cpu().tolist(),
                                        }
                                else:
                                    # No valid kernels above threshold - use empty tensor
                                    kernel_values_list[b].append(torch.tensor([], device=device, dtype=chunk_kernel_b.dtype))
                                    gaussian_indices_list[b].append(torch.tensor([], device=device, dtype=torch.long))
                                    num_nonzero[b, n] = 0
                                    if getattr(self, '_debug_point_indices', None) is not None and n in self._debug_point_indices:
                                        self._debug_capture[n] = {'gaussian_indices': [], 'kernel_values': []}
                            
                            # Free intermediate tensors immediately
                            del chunk_kernel_b_masked, topk_values, topk_indices, valid_mask
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            # No max_kernels_per_point limit: process each query point
                            # Optimized: use torch.nonzero with as_tuple=True for better performance
                            chunk_N_actual = chunk_end - chunk_start
                            for n_offset in range(chunk_N_actual):
                                n = chunk_start + n_offset
                                point_mask = chunk_mask_b[n_offset, :]  # [G]
                                
                                if point_mask.any():
                                    # Use boolean indexing directly (more efficient)
                                    point_kernel_values = chunk_kernel_b[n_offset, point_mask]  # [K_n]
                                    # Use nonzero with as_tuple=True for better performance when extracting indices
                                    point_gaussian_indices = point_mask.nonzero(as_tuple=False).squeeze(-1)  # [K_n]
                                    
                                    kernel_values_list[b].append(point_kernel_values)
                                    gaussian_indices_list[b].append(point_gaussian_indices)
                                    num_nonzero[b, n] = len(point_kernel_values)
                                else:
                                    kernel_values_list[b].append(torch.tensor([], device=device, dtype=chunk_kernel_b.dtype))
                                    gaussian_indices_list[b].append(torch.tensor([], device=device, dtype=torch.long))
                                    num_nonzero[b, n] = 0
                            
                            # Free intermediate tensors
                            del chunk_mask_b
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                else:
                    # Accumulate to dense tensor
                    kernel_values[:, chunk_start:chunk_end, :] = chunk_kernel_values
                
                # Free chunk tensors
                del chunk_distances, chunk_kernel_values, chunk_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if self.verbose:
                total = B * N * G
                percentage = (total_filtered / total) * 100
                print(f"[AnisotropicKernel] Distance filtering: {total_filtered}/{total} "
                      f"({percentage:.2f}%) Gaussians within {self.max_euclidean_distance}m")
                if sparsify_on_the_fly:
                    mean_nonzero = num_nonzero.float().mean().item()
                    max_nonzero = num_nonzero.max().item()
                    print(f"[AnisotropicKernel] On-the-fly sparsification: mean_nonzero={mean_nonzero:.1f}, max_nonzero={max_nonzero}")
                print(f"[AnisotropicKernel] Streaming chunking completed, peak memory saved")
            
            # Free cov_inv and means_expanded (no longer needed)
            del cov_inv, means_expanded, ell_expanded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return sparse format if sparsify_on_the_fly was used
            if sparsify_on_the_fly:
                if self.timing_profiler is not None:
                    self.timing_profiler.end('compute_kernel_batch')
                return {
                    'kernel_values': kernel_values_list,  # List[List[Tensor]]
                    'gaussian_indices': gaussian_indices_list,  # List[List[Tensor]]
                    'num_nonzero': num_nonzero,  # [B, N]
                    'threshold': sparsify_threshold,
                    'max_kernels_per_point': sparsify_max_kernels_per_point,
                    'device': device,
                    'dtype': query_points.dtype
                }
        
        else:
            # Non-streaming mode: compute everything at once (for small N)
            # This preserves backward compatibility for small batches
            distance_mask = None
            
            if self.max_euclidean_distance is not None and self.max_euclidean_distance > 0:
                # Compute Euclidean distances: [B, N, G]
                query_expanded = query_points.unsqueeze(2)  # [B, N, 1, 3]
                means_expanded = gaussian_means.unsqueeze(1)  # [B, 1, G, 3]
                diff = query_expanded - means_expanded  # [B, N, G, 3]
                euclidean_distances = torch.norm(diff, dim=-1)  # [B, N, G]
                distance_mask = euclidean_distances <= self.max_euclidean_distance  # [B, N, G]
                
                if self.verbose:
                    num_within_distance = distance_mask.sum().item()
                    total = B * N * G
                    percentage = (num_within_distance / total) * 100
                    print(f"[AnisotropicKernel] Distance filtering: {num_within_distance}/{total} "
                          f"({percentage:.2f}%) Gaussians within {self.max_euclidean_distance}m")
            
            # Compute distances
            # Optimization: Filter by Euclidean distance BEFORE computing expensive Mahalanobis distance
            if use_ellipsoid_distance:
                distances = self.compute_ellipsoid_surface_distance(
                    query_points, gaussian_means, gaussian_covariances, tau=tau
                )  # [B, N, G]
                if distance_mask is not None:
                    distances = torch.where(distance_mask, distances,
                                           torch.tensor(float('inf'), device=device, dtype=query_points.dtype))
            else:
                # Compute Mahalanobis distance
                # First compute Euclidean distance to filter out distant Gaussians
                query_expanded = query_points.unsqueeze(2)  # [B, N, 1, 3]
                means_expanded = gaussian_means.unsqueeze(1)  # [B, 1, G, 3]
                diff = query_expanded - means_expanded  # [B, N, G, 3]
                
                # Initialize distances with inf (for Gaussians beyond max_euclidean_distance)
                distances = torch.full((B, N, G), float('inf'), 
                                     device=device, dtype=query_points.dtype)
                
                # Only compute Mahalanobis distance for Gaussians within max_euclidean_distance
                if distance_mask is None or distance_mask.any():
                    eye = torch.eye(3, dtype=gaussian_covariances.dtype, device=device)
                    cov_reg = gaussian_covariances + self.eps * eye.view(1, 1, 3, 3)  # [B, G, 3, 3]
                    
                    try:
                        cov_inv = torch.inverse(cov_reg)  # [B, G, 3, 3]
                    except RuntimeError:
                        if self.verbose:
                            dets = torch.det(cov_reg)
                            singular_mask = torch.abs(dets) < 1e-6
                            if singular_mask.any():
                                num_singular = singular_mask.sum().item()
                                print(f"[AnisotropicKernel] Error: {num_singular} singular covariance matrices detected in batch")
                        cov_inv = torch.pinverse(cov_reg)  # [B, G, 3, 3]
                    
                    cov_inv_expanded = cov_inv.unsqueeze(1)  # [B, 1, G, 3, 3]
                    diff_expanded = diff.unsqueeze(-1)  # [B, N, G, 3, 1]
                    cov_inv_diff = torch.matmul(cov_inv_expanded, diff_expanded)  # [B, N, G, 3, 1]
                    cov_inv_diff = cov_inv_diff.squeeze(-1)  # [B, N, G, 3]
                    mahalanobis_dist_sq = torch.sum(diff * cov_inv_diff, dim=-1)  # [B, N, G]
                    mahalanobis_dist_sq = torch.clamp(mahalanobis_dist_sq, min=0.0, max=self.max_mahalanobis_dist ** 2)
                    mahalanobis_dist = torch.sqrt(mahalanobis_dist_sq + self.eps)  # [B, N, G]
                    
                    if distance_mask is not None:
                        # Only keep Mahalanobis distances for Gaussians within max_euclidean_distance
                        distances = torch.where(distance_mask, mahalanobis_dist, distances)
                    else:
                        distances = mahalanobis_dist
            
            # Compute kernel values
            if self.use_paper_kernel:
                ell_expanded = ell.unsqueeze(1)  # [B, 1, G]
                kernel_values = self.k_prime(distances, ell_expanded)  # [B, N, G]
            else:
                dist_sq = distances ** 2
                dist_sq = torch.clamp(dist_sq, min=0.0, max=self.max_mahalanobis_dist ** 2)
                if ell.shape == (B, G):
                    ell_for_kernel = ell.unsqueeze(1)  # [B, 1, G]
                    kernel_values = torch.exp(-0.5 * dist_sq) * ell_for_kernel  # [B, N, G]
                else:
                    kernel_values = torch.exp(-0.5 * dist_sq) * self.scale  # [B, N, G]
            
            # Apply distance mask: set kernel values to 0 for Gaussians outside max_euclidean_distance
            if distance_mask is not None:
                kernel_values = kernel_values * distance_mask.float()  # [B, N, G]
        
        # Validate output (only for dense format, sparse format is validated during sparsification)
        # Note: kernel_values may be sparse (dict) if sparsify_on_the_fly was used
        if isinstance(kernel_values, dict):
            # Sparse format: skip validation (already validated during sparsification)
            if self.timing_profiler is not None:
                self.timing_profiler.end('compute_kernel_batch')
            return kernel_values
        
        # Dense format validation
        # For large tensors, avoid creating full masks (can cause OOM)
        # Use chunked validation for large tensors
        total_elements = kernel_values.numel()
        if total_elements > 1e8:  # > 100M elements: use chunked validation
            # For very large tensors, validate in chunks to avoid OOM
            chunk_size = min(4096, kernel_values.shape[1])  # Use same chunk size as computation
            has_nan = False
            has_inf = False
            num_nan = 0
            num_inf = 0
            
            for i in range(0, kernel_values.shape[1], chunk_size):
                chunk = kernel_values[:, i:i+chunk_size, :]
                chunk_nan = torch.isnan(chunk).any().item()
                chunk_inf = torch.isinf(chunk).any().item()
                if chunk_nan:
                    has_nan = True
                    num_nan += torch.isnan(chunk).sum().item()
                if chunk_inf:
                    has_inf = True
                    num_inf += torch.isinf(chunk).sum().item()
                
                # Free chunk immediately
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if has_nan:
                raise ValueError(f"kernel_values contains {num_nan} NaN values (tensor too large: {total_elements} elements, validated in chunks)")
            
            # For masked Gaussians, we expect some Inf values, so only warn if too many
            inf_ratio = num_inf / total_elements
            if inf_ratio > 0.9:  # More than 90% are Inf, likely an error
                raise ValueError(f"kernel_values contains too many ({num_inf}/{total_elements}, {inf_ratio*100:.1f}%) Inf values")
        else:
            # For smaller tensors, use normal validation
            has_nan = torch.isnan(kernel_values).any().item()
            has_inf = torch.isinf(kernel_values).any().item()
            
            if has_nan:
                num_nan = torch.isnan(kernel_values).sum().item()
                if total_elements < 2**31:  # INT_MAX
                    try:
                        nan_mask = torch.isnan(kernel_values)
                        nan_indices = torch.nonzero(nan_mask, as_tuple=False)[:10]
                        raise ValueError(f"kernel_values contains {num_nan} NaN values. First 10 indices: {nan_indices.tolist()}")
                    except RuntimeError:
                        raise ValueError(f"kernel_values contains {num_nan} NaN values (unable to extract indices)")
                else:
                    raise ValueError(f"kernel_values contains {num_nan} NaN values (tensor too large: {total_elements} elements)")
            
            if has_inf:
                num_inf = torch.isinf(kernel_values).sum().item()
                inf_ratio = num_inf / total_elements
                if inf_ratio > 0.9:  # More than 90% are Inf, likely an error
                    if total_elements < 2**31:  # INT_MAX
                        try:
                            inf_mask = torch.isinf(kernel_values)
                            inf_indices = torch.nonzero(inf_mask, as_tuple=False)[:10]
                            raise ValueError(f"kernel_values contains too many ({num_inf}, {inf_ratio*100:.1f}%) Inf values. First 10 indices: {inf_indices.tolist()}")
                        except RuntimeError:
                            raise ValueError(f"kernel_values contains too many ({num_inf}, {inf_ratio*100:.1f}%) Inf values (unable to extract indices)")
                    else:
                        raise ValueError(f"kernel_values contains too many ({num_inf}, {inf_ratio*100:.1f}%) Inf values (tensor too large: {total_elements} elements)")
        
        if self.verbose:
            # For large tensors, check in chunks to avoid OOM
            total_elements = kernel_values.numel()
            if total_elements > 1e8:  # > 100M elements: use chunked check
                has_negative = False
                has_too_large = False
                num_invalid = 0
                chunk_size = min(4096, kernel_values.shape[1])
                
                for i in range(0, kernel_values.shape[1], chunk_size):
                    chunk = kernel_values[:, i:i+chunk_size, :]
                    chunk_negative = (chunk < 0).any().item()
                    chunk_too_large = (chunk > self.scale * 1.1).any().item()
                    if chunk_negative or chunk_too_large:
                        has_negative = has_negative or chunk_negative
                        has_too_large = has_too_large or chunk_too_large
                        invalid_chunk = (chunk < 0) | (chunk > self.scale * 1.1)
                        num_invalid += invalid_chunk.sum().item()
                    del chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if has_negative or has_too_large:
                    print(f"[AnisotropicKernel] Warning: {num_invalid} kernel values out of expected range [0, {self.scale}]")
            else:
                # For smaller tensors, use normal check
                if (kernel_values < 0).any() or (kernel_values > self.scale * 1.1).any():
                    invalid_mask = (kernel_values < 0) | (kernel_values > self.scale * 1.1)
                    num_invalid = invalid_mask.sum().item()
                    invalid_values = kernel_values[invalid_mask]
                    print(f"[AnisotropicKernel] Warning: {num_invalid} kernel values out of expected range [0, {self.scale}]")
                    print(f"  Invalid values sample: {invalid_values.tolist()[:10]}")
        
        if self.timing_profiler is not None:
            self.timing_profiler.end('compute_kernel_batch')
        return kernel_values
    
    def sparsify_kernel_values(self, kernel_values, threshold=1e-6, max_kernels_per_point=None):
        """
        将kernel values稀疏化，只保留大于阈值的值，并限制每个query point的最大kernel数量
        
        使用COO格式（Coordinate Format）存储稀疏kernel值：
        - 为每个query point存储非零kernel值和对应的Gaussian索引
        - 限制每个query point的最大kernel数量，防止显存溢出
        
        Args:
            kernel_values: [B, N, G] - 完整kernel值矩阵
            threshold: float - 阈值，只保留kernel值 > threshold的值（默认：1e-6）
            max_kernels_per_point: int or None - 每个query point的最大kernel数量上限（默认：None，无限制）
                如果超过上限，保留kernel值最大的max_kernels_per_point个
        
        Returns:
            sparse_kernel: dict - 稀疏kernel数据结构
                - kernel_values: List[List[Tensor]] - [B]个列表，每个列表包含[N]个tensor
                    每个tensor包含该query point的非零kernel值 [K_n]
                - gaussian_indices: List[List[Tensor]] - [B]个列表，每个列表包含[N]个tensor
                    每个tensor包含对应的Gaussian索引 [K_n]（long类型）
                - num_nonzero: Tensor [B, N] - 每个query point的非零kernel数量
                - threshold: float - 使用的阈值
                - max_kernels_per_point: int or None - 使用的最大kernel数量上限
        """
        if kernel_values is None:
            raise ValueError("kernel_values cannot be None")
        
        if len(kernel_values.shape) != 3:
            raise ValueError(f"kernel_values must be [B, N, G], got {kernel_values.shape}")
        
        B, N, G = kernel_values.shape
        device = kernel_values.device
        dtype = kernel_values.dtype
        
        # 为每个query point收集非零kernel值和索引
        # 使用chunked处理避免创建完整mask tensor导致OOM
        kernel_values_list = []
        gaussian_indices_list = []
        num_nonzero = torch.zeros(B, N, device=device, dtype=torch.long)
        
        # 对于大tensor，使用chunked处理避免创建完整mask
        total_elements = kernel_values.numel()
        use_chunked_sparsify = total_elements > 1e8  # > 100M elements
        
        if use_chunked_sparsify:
            # Chunked processing: process query points in chunks
            chunk_size = 4096  # Same as streaming chunking size
            if self.verbose:
                print(f"[AnisotropicKernel] Using chunked sparsification: N={N}, chunk_size={chunk_size}")
            
            for b in range(B):
                batch_kernel_values = []
                batch_gaussian_indices = []
                
                # Process query points in chunks
                for chunk_start in range(0, N, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, N)
                    chunk_kernel = kernel_values[b, chunk_start:chunk_end, :]  # [chunk_N, G]
                    
                    # Create mask for this chunk only
                    chunk_mask = chunk_kernel > threshold  # [chunk_N, G]
                    
                    # Process each query point in this chunk
                    for n_offset in range(chunk_end - chunk_start):
                        n = chunk_start + n_offset
                        point_mask = chunk_mask[n_offset, :]  # [G]
                        point_kernel_values = chunk_kernel[n_offset, :][point_mask]  # [K_n]
                        point_gaussian_indices = torch.nonzero(point_mask, as_tuple=False).squeeze(-1)  # [K_n]
                        
                        # 如果设置了max_kernels_per_point，只保留kernel值最大的前K个
                        if max_kernels_per_point is not None and len(point_kernel_values) > max_kernels_per_point:
                            # 按kernel值降序排序，取前max_kernels_per_point个
                            _, top_indices = torch.topk(point_kernel_values, max_kernels_per_point, largest=True)
                            point_kernel_values = point_kernel_values[top_indices]
                            point_gaussian_indices = point_gaussian_indices[top_indices]
                        
                        batch_kernel_values.append(point_kernel_values)
                        batch_gaussian_indices.append(point_gaussian_indices)
                        num_nonzero[b, n] = len(point_kernel_values)
                    
                    # Free chunk tensors
                    del chunk_kernel, chunk_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                kernel_values_list.append(batch_kernel_values)
                gaussian_indices_list.append(batch_gaussian_indices)
        else:
            # Normal processing for smaller tensors
            # 创建mask: [B, N, G]
            mask = kernel_values > threshold
            
            for b in range(B):
                batch_kernel_values = []
                batch_gaussian_indices = []
                
                for n in range(N):
                    # 获取当前query point的非零kernel值和索引
                    point_mask = mask[b, n, :]  # [G]
                    point_kernel_values = kernel_values[b, n, :][point_mask]  # [K_n]
                    point_gaussian_indices = torch.nonzero(point_mask, as_tuple=False).squeeze(-1)  # [K_n]
                    
                    # 如果设置了max_kernels_per_point，只保留kernel值最大的前K个
                    if max_kernels_per_point is not None and len(point_kernel_values) > max_kernels_per_point:
                        # 按kernel值降序排序，取前max_kernels_per_point个
                        _, top_indices = torch.topk(point_kernel_values, max_kernels_per_point, largest=True)
                        point_kernel_values = point_kernel_values[top_indices]
                        point_gaussian_indices = point_gaussian_indices[top_indices]
                    
                    batch_kernel_values.append(point_kernel_values)
                    batch_gaussian_indices.append(point_gaussian_indices)
                    num_nonzero[b, n] = len(point_kernel_values)
                
                kernel_values_list.append(batch_kernel_values)
                gaussian_indices_list.append(batch_gaussian_indices)
        
        return {
            'kernel_values': kernel_values_list,  # List[List[Tensor]]
            'gaussian_indices': gaussian_indices_list,  # List[List[Tensor]]
            'num_nonzero': num_nonzero,  # [B, N]
            'threshold': threshold,
            'max_kernels_per_point': max_kernels_per_point,
            'device': device,
            'dtype': dtype
        }
    
    def forward(self, query_points, gaussian_means, gaussian_scales, gaussian_rotations):
        """
        Forward pass: compute kernel values from Gaussian parameters.
        
        This is a convenience method that computes covariance matrices internally.
        
        Args:
            query_points: [N, 3] or [B, N, 3] - Query points
            gaussian_means: [G, 3] or [B, G, 3] - Gaussian centers
            gaussian_scales: [G, 3] or [B, G, 3] - Gaussian scales
            gaussian_rotations: [G, 4] or [B, G, 4] - Rotation quaternions
        
        Returns:
            kernel_values: [N, G] or [B, N, G] - Kernel values
        """
        # Input validation
        if query_points is None or gaussian_means is None:
            raise ValueError("query_points and gaussian_means cannot be None")
        
        if gaussian_scales is None or gaussian_rotations is None:
            raise ValueError("gaussian_scales and gaussian_rotations cannot be None")
        
        # Validate input shapes (must be batched)
        if len(query_points.shape) != 3 or query_points.shape[2] != 3:
            raise ValueError(f"query_points must be [B, N, 3], got {query_points.shape}")
        if len(gaussian_means.shape) != 3 or gaussian_means.shape[2] != 3:
            raise ValueError(f"gaussian_means must be [B, G, 3], got {gaussian_means.shape}")
        if query_points.shape[0] != gaussian_means.shape[0]:
            raise ValueError(f"Batch size mismatch: query_points B={query_points.shape[0]}, "
                           f"gaussian_means B={gaussian_means.shape[0]}")
        
        # Compute covariance matrices (with validation)
        covariances = self.compute_covariance_matrix(gaussian_scales, gaussian_rotations)  # [B, G, 3, 3]
        
        # Compute kernel values (batch version)
        return self.compute_kernel_batch(query_points, gaussian_means, covariances)  # [B, N, G]

