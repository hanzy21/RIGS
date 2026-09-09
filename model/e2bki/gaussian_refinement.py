"""
Gaussian Refinement for E2-BKI

This module implements Gaussian refinement through pruning and merging according to the paper.

Reference:
    Kim et al., "E2-BKI: Evidential Ellipsoidal Bayesian Kernel Inference", 2025
    Section IV-C: Gaussian Refinement

Algorithm:
    - Merging: Check semantic consistency in dL radius, merge primitives in dS radius
              Use MoM (Method of Moments) for geometric components
              Use DST-based combination rules for semantic components
    - Pruning: Relative pruning strategy based on sensor distance δj
              Prune only when neighbor has lower δi and conflicting semantics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.spatial import cKDTree
from ..encoder.gaussian_encoder.utils import GaussianPrediction


class GaussianRefinement(nn.Module):
    """
    Gaussian Refinement: Pruning and Merging
    
    - Merging: combines nearby primitives into one representation
    
    Args:
        kernel_scale (float): Kernel scale ℓ in meters (default: 0.2m, from Table I)
        dL (float): Large radius for semantic consistency check (default: 5ℓ = 1.0m)
        dS (float): Small radius for merging (default: ℓ = 0.2m)
        epsilon (float): Pruning sensitivity parameter ϵ (default: 2.5, from Table I)
        device (str): Device for computation (default: 'cuda')
        verbose (bool): Whether to print detailed debug information (default: False)
    """
    
    def __init__(self, 
                 kernel_scale=0.2,
                 dL=None,
                 dS=None,
                 epsilon=2.5,
                 device='cuda',
                 verbose=False):
        super().__init__()
        self.kernel_scale = kernel_scale  # ℓ
        self.dL = dL if dL is not None else 5 * kernel_scale  # 5ℓ
        self.dS = dS if dS is not None else kernel_scale  # ℓ
        self.epsilon = epsilon  # ϵ
        self.device = device
        self.verbose = verbose
    
    def _dst_combine(self, p1, u1, p2, u2, num_classes):
        """
        DST-based combination rule (Equation 5 in paper).
        
        Combines two belief masses and converts back to probability.
        
        Args:
            p1: [C] - Semantic probability distribution 1
            u1: scalar - Uncertainty 1
            p2: [C] - Semantic probability distribution 2
            u2: scalar - Uncertainty 2
            num_classes: int - Number of semantic classes C
        
        Returns:
            p_merged: [C] - Merged semantic probability
            u_merged: scalar - Merged uncertainty
        """
        # Convert probabilities to belief masses: b^c = p^c - u/C
        b1 = p1 - u1 / num_classes  # [C]
        b2 = p2 - u2 / num_classes  # [C]
        
        # Compute conflict measure: δ = Σ_{x≠y} b1^x * b2^y
        # This is the sum of all cross-products where x != y
        # Efficient computation: δ = sum(b1) * sum(b2) - sum(b1 * b2)
        # But we need to exclude diagonal terms, so:
        # δ = sum_{x≠y} b1[x] * b2[y] = (sum(b1) * sum(b2)) - sum(b1 * b2)
        # However, this includes x=y terms, so we subtract them:
        delta = torch.sum(b1) * torch.sum(b2) - torch.dot(b1, b2)
        delta = delta.item()
        
        # Avoid division by zero
        if abs(1.0 - delta) < 1e-8:
            # High conflict, use simple weighted average as fallback
            w1 = 1.0 / (u1 + 1e-8)
            w2 = 1.0 / (u2 + 1e-8)
            w_sum = w1 + w2
            p_merged = (w1 * p1 + w2 * p2) / w_sum
            u_merged = (w1 * u1 + w2 * u2) / w_sum
            return p_merged, u_merged
        
        # DST combination rule
        # b^c = (1/(1-δ)) * (b1^c * b2^c + b1^c * u2 + b2^c * u1)
        # u = (1/(1-δ)) * u1 * u2
        denom = 1.0 - delta
        b_merged = (b1 * b2 + b1 * u2 + b2 * u1) / denom
        u_merged = (u1 * u2) / denom
        
        # Convert back to probability: p^c = b^c + u/C
        p_merged = b_merged + u_merged / num_classes
        
        # Ensure valid probability distribution
        p_merged = torch.clamp(p_merged, 0.0, 1.0)
        p_merged = p_merged / (p_merged.sum() + 1e-8)  # Normalize
        
        return p_merged, u_merged
    
    def _mom_merge_geometric(self, mean1, cov1, mean2, cov2, eta1=1.0, eta2=1.0):
        """
        Merge geometric components using Method of Moments (MoM) according to paper Equation 26.
        
        MoM representation maintains:
        - First moment: m^(1) = η * μ
        - Second moment: M^(2) = η * (Σ + μμ^T)
        - Normalization constant: η
        
        Merging: accumulate moments and normalization constants.
        
        Args:
            mean1: [3] - Mean of Gaussian 1
            cov1: [3, 3] - Covariance of Gaussian 1
            mean2: [3] - Mean of Gaussian 2
            cov2: [3, 3] - Covariance of Gaussian 2
            eta1: scalar - Normalization constant for Gaussian 1 (default: 1.0)
            eta2: scalar - Normalization constant for Gaussian 2 (default: 1.0)
        
        Returns:
            merged_mean: [3] - Merged mean
            merged_cov: [3, 3] - Merged covariance
            merged_eta: scalar - Merged normalization constant
        """
        # Convert to MoM representation
        # First moment: m^(1) = η * μ
        m1_first = eta1 * mean1  # [3]
        m2_first = eta2 * mean2  # [3]
        
        # Second moment: M^(2) = η * (Σ + μμ^T)
        m1_second = eta1 * (cov1 + torch.outer(mean1, mean1))  # [3, 3]
        m2_second = eta2 * (cov2 + torch.outer(mean2, mean2))  # [3, 3]
        
        # Accumulate moments
        merged_eta = eta1 + eta2
        merged_m_first = m1_first + m2_first  # [3]
        merged_m_second = m1_second + m2_second  # [3, 3]
        
        # Convert back to Gaussian parameters (Equation 26)
        # μ = (1/η) * m^(1)
        merged_mean = merged_m_first / merged_eta
        
        # Σ = (1/η) * M^(2) - μμ^T
        merged_cov = merged_m_second / merged_eta - torch.outer(merged_mean, merged_mean)
        
        return merged_mean, merged_cov, merged_eta
    
    def _cov_to_scale_rotation(self, cov):
        """
        Convert covariance matrix back to scale and rotation.
        
        This is the inverse of computing covariance from scale and rotation.
        We use eigenvalue decomposition: Cov = R * diag(scale^2) * R^T
        
        Args:
            cov: [3, 3] - Covariance matrix
        
        Returns:
            scale: [3] - Scale vector
            rotation: [4] - Quaternion rotation
        """
        # Eigenvalue decomposition
        eigenvals, eigenvecs = torch.linalg.eigh(cov)
        
        # Scale is square root of eigenvalues
        scale = torch.sqrt(torch.clamp(eigenvals, min=1e-8))
        
        # Rotation matrix is the eigenvector matrix
        # Convert rotation matrix to quaternion
        R = eigenvecs
        
        # Convert rotation matrix to quaternion
        # Using standard conversion formula
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        rotation = torch.stack([w, x, y, z])
        rotation = F.normalize(rotation, dim=0)  # Normalize quaternion
        
        return scale, rotation
    
    def compute_sensor_distances(self, gaussian_means, sensor_positions):
        """
        Compute sensor distance δj for each Gaussian.
        
        δj = average distance from sensor to points aggregated in primitive.
        Since we only have Gaussian means, we use distance from sensor to mean.
        
        IMPORTANT: Each Gaussian should have its own sensor_position (the sensor position
        when that Gaussian was detected), not a fixed position. This allows pruning based
        on relative sensor distances across different frames.
        
        Args:
            gaussian_means: [B, G, 3] - Gaussian means (batched)
            sensor_positions: [B, G, 3] or [B, 3] - Sensor position(s) (batched)
                - [B, G, 3]: Each Gaussian has its own sensor position (recommended)
                - [B, 3]: Single sensor position for all Gaussians (fallback)
        
        Returns:
            sensor_distances: [B, G] - Sensor distances
        """
        if len(gaussian_means.shape) != 3 or gaussian_means.shape[2] != 3:
            raise ValueError(f"gaussian_means must be [B, G, 3], got {gaussian_means.shape}")
        
        # [B, G, 3] - batched
        B, G, _ = gaussian_means.shape
        
        if len(sensor_positions.shape) == 2:
            if sensor_positions.shape == (B, 3):
                # [B, 3] - single sensor position per batch (fallback)
                diff = gaussian_means - sensor_positions.unsqueeze(1)  # [B, G, 3]
                distances = torch.norm(diff, dim=2)  # [B, G]
            else:
                raise ValueError(f"sensor_positions must be [B, 3] or [B, G, 3], got {sensor_positions.shape}")
        elif len(sensor_positions.shape) == 3:
            if sensor_positions.shape == (B, G, 3):
                # [B, G, 3] - each Gaussian has its own sensor position (recommended)
                diff = gaussian_means - sensor_positions  # [B, G, 3]
                distances = torch.norm(diff, dim=2)  # [B, G]
            else:
                raise ValueError(f"sensor_positions shape {sensor_positions.shape} not compatible with gaussian_means shape {gaussian_means.shape}")
        else:
            raise ValueError(f"sensor_positions must be [B, 3] or [B, G, 3], got {sensor_positions.shape}")
        
        return distances
    
        #TODO: sensor_positions应该是检测到这个高斯的那一帧时，高斯到传感器的位置，后续车辆前进，在相近空间点检测到新的高斯后再做剪枝。这里目前还没有读取真实的sensor_positions。
    def prune_gaussians(self, gaussians, sensor_positions):
        """
        Relative pruning strategy according to paper Section IV-C.
        
        Prune primitive Gj only when:
        1. A neighboring primitive Gi has significantly lower sensor distance δi
        2. Gi and Gj have conflicting semantics: argmax pi ≠ argmax pj
        3. δj > ϵ · δi (where ϵ = 2.5)
        
        IMPORTANT: sensor_positions should be the sensor position when each Gaussian was detected,
        not a fixed position. This allows pruning based on relative sensor distances across different frames.
        For example, when the sensor moves closer to a target and detects a more reliable Gi,
        it can prune a previously detected unreliable Gj.
        
        Args:
            gaussians: GaussianPrediction object (must be batched)
            sensor_positions: [B, G, 3] or [B, 3] - Sensor position(s) (batched)
                - [B, G, 3]: Each Gaussian has its own sensor position (recommended)
                - [B, 3]: Single sensor position for all Gaussians (fallback, less accurate)
        
        Returns:
            pruned_gaussians: Pruned GaussianPrediction object (batched)
            mask: [B, G] - Retention mask (True means keep)
        """
        if gaussians is None:
            raise ValueError("gaussians cannot be None")
        if sensor_positions is None:
            raise ValueError("sensor_positions cannot be None")
        
        # Validate input shapes (must be batched)
        if len(gaussians.means.shape) != 3 or gaussians.means.shape[2] != 3:
            raise ValueError(f"gaussians.means must be [B, G, 3], got {gaussians.means.shape}")
        
        B, G, _ = gaussians.means.shape
        
        # Ensure sensor_positions shape matches: [B, G, 3] or [B, 3]
        if len(sensor_positions.shape) == 2:
            if sensor_positions.shape == (B, 3):
                # [B, 3] - broadcast to [B, G, 3] for compatibility (fallback)
                sensor_positions = sensor_positions.unsqueeze(1).expand(B, G, 3)
            else:
                raise ValueError(f"sensor_positions shape {sensor_positions.shape} not compatible with gaussians.means shape {gaussians.means.shape}")
        elif len(sensor_positions.shape) == 3:
            if sensor_positions.shape == (B, G, 3):
                # [B, G, 3] - already correct (recommended)
                pass
            else:
                raise ValueError(f"sensor_positions shape {sensor_positions.shape} not compatible with gaussians.means shape {gaussians.means.shape}")
        else:
            raise ValueError(f"sensor_positions must be [B, 3] or [B, G, 3], got {sensor_positions.shape}")
        
        # Compute sensor distances (each Gaussian uses its own sensor position)
        sensor_distances = self.compute_sensor_distances(gaussians.means, sensor_positions)  # [B, G]
        
        # Get predicted semantic classes (argmax)
        if len(gaussians.semantics.shape) == 3:
            # [B, G, C]
            predicted_classes = torch.argmax(gaussians.semantics, dim=2)  # [B, G]
        else:
            raise ValueError(f"gaussians.semantics shape must be [B, G, C], got {gaussians.semantics.shape}")
        
        # Query local neighbours on CPU. The previous implementation compared
        # every pair in Python (O(G^2)) and never applied the documented dL
        # spatial-neighbour condition.
        mask_np = np.ones((B, G), dtype=bool)
        for b in range(B):
            means_b = gaussians.means[b].detach().cpu().numpy()
            distances_b = sensor_distances[b].detach().cpu().numpy()
            classes_b = predicted_classes[b].detach().cpu().numpy()
            neighbours = cKDTree(means_b).query_ball_point(means_b, r=self.dL)
            for i in range(G):
                if not mask_np[b, i]:
                    continue
                candidate = np.asarray(neighbours[i], dtype=np.int64)
                candidate = candidate[candidate != i]
                if candidate.size == 0:
                    continue
                prune = (
                    mask_np[b, candidate]
                    & (classes_b[candidate] != classes_b[i])
                    & (distances_b[candidate] > self.epsilon * distances_b[i])
                )
                mask_np[b, candidate[prune]] = False
        mask = torch.as_tensor(mask_np, device=gaussians.means.device)
        
        # Count pruned Gaussians
        num_pruned = (~mask).sum().item()
        num_total = mask.numel()
        pruned_ratio = num_pruned / num_total if num_total > 0 else 0.0
        
        if self.verbose:
            print(f"[GaussianRefinement] Pruning: {num_pruned}/{num_total} Gaussians removed ({pruned_ratio*100:.1f}%)")
        
        # Check if all Gaussians are pruned
        if mask.sum().item() == 0:
            raise ValueError(f"All Gaussians pruned! Consider adjusting epsilon (current: {self.epsilon})")
        
        # Apply mask
        pruned_gaussians = self._apply_mask(gaussians, mask)
        
        return pruned_gaussians, mask
    
    def merge_gaussians(self, gaussians):
        """
        Merge nearby primitives according to paper Section IV-C.
        
        Algorithm:
        1. For each Gaussian Gj, check all neighbors within radius dL
        2. If all neighbors within dL belong to the same semantic class (semantic consistency)
        3. Merge primitives within dS radius
        4. Use MoM for geometric components, DST for semantic components
        
        Args:
            gaussians: GaussianPrediction object (must be batched)
        
        Returns:
            merged_gaussians: Merged GaussianPrediction object (batched)
        """
        if gaussians is None:
            raise ValueError("gaussians cannot be None")
        
        # Validate input shapes (must be batched)
        if len(gaussians.means.shape) != 3 or gaussians.means.shape[2] != 3:
            raise ValueError(f"gaussians.means must be [B, G, 3], got {gaussians.means.shape}")
        
        B, G, _ = gaussians.means.shape
        num_classes = gaussians.semantics.shape[-1]
        
        # Get predicted semantic classes
        predicted_classes = torch.argmax(gaussians.semantics, dim=2)  # [B, G]
        
        # Compute covariance matrices for distance calculation
        from .anisotropic_kernel import AnisotropicKernel
        kernel = AnisotropicKernel(verbose=self.verbose)
        covariances = kernel.compute_covariance_matrix(gaussians.scales, gaussians.rotations)  # [B, G, 3, 3]
        
        # Process each batch
        batch_merged_gaussians = []
        for b in range(B):
            means_b = gaussians.means[b]  # [G, 3]
            scales_b = gaussians.scales[b]  # [G, 3]
            rotations_b = gaussians.rotations[b]  # [G, 4]
            opacities_b = gaussians.opacities[b]  # [G, ...]
            semantics_b = gaussians.semantics[b]  # [G, C]
            velocities_b = gaussians.velocities[b] if gaussians.velocities is not None else None
            alphas_b = gaussians.evidential_alphas[b] if gaussians.evidential_alphas is not None else None
            uncertainties_b = gaussians.uncertainties[b] if gaussians.uncertainties is not None else torch.zeros(G, device=means_b.device)  # [G]
            covariances_b = covariances[b]  # [G, 3, 3]
            classes_b = predicted_classes[b]  # [G]
            
            # Spatial indexing reduces the two documented radius queries from
            # all-pairs Python loops to local-neighbour lookups.
            means_np = means_b.detach().cpu().numpy()
            classes_np = classes_b.detach().cpu().numpy()
            tree = cKDTree(means_np)
            neighbours_dL_all = tree.query_ball_point(means_np, r=self.dL)
            neighbours_dS_all = tree.query_ball_point(means_np, r=self.dS)
            merged = np.zeros(G, dtype=bool)
            merged_list = []
            
            # For each Gaussian i
            for i in range(G):
                if merged[i]:
                    continue
                
                neighbors_dL = [
                    j for j in neighbours_dL_all[i]
                    if j != i and not merged[j]
                ]
                neighbors_dS = [
                    j for j in neighbours_dS_all[i]
                    if j != i and not merged[j]
                ]
                
                # Check semantic consistency in dL neighborhood
                if len(neighbors_dL) > 0:
                    class_i = classes_np[i]
                    all_same_class = all(classes_np[j] == class_i for j in neighbors_dL)
                    
                    if not all_same_class:
                        # Semantic inconsistency in dL, skip merging
                        merged_list.append({
                            'mean': means_b[i],
                            'scale': scales_b[i],
                            'rotation': rotations_b[i],
                            'opacity': opacities_b[i],
                            'semantic': semantics_b[i],
                            'uncertainty': uncertainties_b[i],
                            'velocity': velocities_b[i] if velocities_b is not None else None,
                            'evidential_alpha': alphas_b[i] if alphas_b is not None else None,
                        })
                        merged[i] = True
                        continue
                
                # Merge with neighbors in dS
                if len(neighbors_dS) > 0:
                    # Collect all Gaussians to merge (including i)
                    to_merge = [i] + neighbors_dS
                    
                    # Merge using MoM for geometry, DST for semantics
                    merged_params = self._merge_multiple_gaussians(
                        means_b[to_merge],
                        covariances_b[to_merge],
                        semantics_b[to_merge],
                        uncertainties_b[to_merge],
                        opacities_b[to_merge],
                        velocities_b[to_merge] if velocities_b is not None else None,
                        alphas_b[to_merge] if alphas_b is not None else None,
                        num_classes
                    )
                    
                    merged_list.append(merged_params)
                    
                    # Mark as merged
                    for j in to_merge:
                        merged[j] = True
                    
                    if self.verbose:
                        print(f"[GaussianRefinement] Merged {len(to_merge)} Gaussians (indices: {to_merge})")
                else:
                    # No neighbors to merge, keep as is
                    merged_list.append({
                        'mean': means_b[i],
                        'scale': scales_b[i],
                        'rotation': rotations_b[i],
                        'opacity': opacities_b[i],
                        'semantic': semantics_b[i],
                        'uncertainty': uncertainties_b[i],
                        'velocity': velocities_b[i] if velocities_b is not None else None,
                        'evidential_alpha': alphas_b[i] if alphas_b is not None else None,
                    })
                    merged[i] = True
            
            if len(merged_list) == 0:
                raise ValueError(f"No Gaussians left after merging in batch {b}")
            
            # Stack results
            batch_means = torch.stack([mg['mean'] for mg in merged_list], dim=0)
            batch_scales = torch.stack([mg['scale'] for mg in merged_list], dim=0)
            batch_rotations = torch.stack([mg['rotation'] for mg in merged_list], dim=0)
            batch_opacities = torch.stack([mg['opacity'] for mg in merged_list], dim=0)
            batch_semantics = torch.stack([mg['semantic'] for mg in merged_list], dim=0)
            batch_uncertainties = torch.stack([mg['uncertainty'] for mg in merged_list], dim=0)
            batch_velocities = (torch.stack([mg['velocity'] for mg in merged_list], dim=0)
                                if velocities_b is not None else None)
            batch_alphas = (torch.stack([mg['evidential_alpha'] for mg in merged_list], dim=0)
                            if alphas_b is not None else None)
            
            batch_merged_gaussians.append({
                'means': batch_means,
                'scales': batch_scales,
                'rotations': batch_rotations,
                'opacities': batch_opacities,
                'semantics': batch_semantics,
                'uncertainties': batch_uncertainties,
                'velocities': batch_velocities,
                'evidential_alphas': batch_alphas,
            })
        
        # Stack all batches
        merged_means = torch.stack([bg['means'] for bg in batch_merged_gaussians], dim=0)
        merged_scales = torch.stack([bg['scales'] for bg in batch_merged_gaussians], dim=0)
        merged_rotations = torch.stack([bg['rotations'] for bg in batch_merged_gaussians], dim=0)
        merged_opacities = torch.stack([bg['opacities'] for bg in batch_merged_gaussians], dim=0)
        merged_semantics = torch.stack([bg['semantics'] for bg in batch_merged_gaussians], dim=0)
        merged_uncertainties = torch.stack([bg['uncertainties'] for bg in batch_merged_gaussians], dim=0)
        merged_velocities = (torch.stack([bg['velocities'] for bg in batch_merged_gaussians], dim=0)
                             if batch_merged_gaussians[0]['velocities'] is not None else None)
        merged_alphas = (torch.stack([bg['evidential_alphas'] for bg in batch_merged_gaussians], dim=0)
                         if batch_merged_gaussians[0]['evidential_alphas'] is not None else None)
        
        merged_gaussians = GaussianPrediction(
            means=merged_means,
            scales=merged_scales,
            rotations=merged_rotations,
            opacities=merged_opacities,
            semantics=merged_semantics,
            uncertainties=merged_uncertainties,
            evidential_alphas=merged_alphas,
            original_means=None,
            delta_means=None,
            velocities=merged_velocities,
        )
        
        return merged_gaussians
    
    def _merge_multiple_gaussians(self, means, covariances, semantics, uncertainties,
                                  opacities, velocities, evidential_alphas, num_classes):
        """
        Merge multiple Gaussians using MoM for geometry and DST for semantics.
        
        Args:
            means: [N, 3] - Means of N Gaussians
            covariances: [N, 3, 3] - Covariances of N Gaussians
            semantics: [N, C] - Semantic probabilities
            uncertainties: [N] - Uncertainties
            opacities: [N, ...] - Opacities
            num_classes: int - Number of semantic classes
        
        Returns:
            merged: dict with merged parameters
        """
        N = means.shape[0]
        
        if N == 1:
            # Single Gaussian, no merging needed
            scale, rotation = self._cov_to_scale_rotation(covariances[0])
            return {
                'mean': means[0],
                'scale': scale,
                'rotation': rotation,
                'opacity': opacities[0],
                'semantic': semantics[0],
                'uncertainty': uncertainties[0],
                'velocity': velocities[0] if velocities is not None else None,
                'evidential_alpha': evidential_alphas[0] if evidential_alphas is not None else None,
            }
        
        # Merge geometry using MoM (iteratively merge pairs)
        # Initialize with first Gaussian (η=1.0)
        merged_mean = means[0]
        merged_cov = covariances[0]
        merged_eta = 1.0
        
        # Iteratively merge remaining Gaussians
        for i in range(1, N):
            merged_mean, merged_cov, merged_eta = self._mom_merge_geometric(
                merged_mean, merged_cov,
                means[i], covariances[i],
                eta1=merged_eta, eta2=1.0
            )
        
        # Convert merged covariance back to scale and rotation
        merged_scale, merged_rotation = self._cov_to_scale_rotation(merged_cov)
        
        # Merge semantics using DST (iteratively combine pairs)
        merged_sem = semantics[0]
        merged_unc = uncertainties[0]
        
        for i in range(1, N):
            merged_sem, merged_unc = self._dst_combine(
                merged_sem, merged_unc,
                semantics[i], uncertainties[i],
                num_classes
            )
        
        # Merge opacities (weighted average)
        if len(opacities.shape) == 1:
            merged_opacity = opacities.mean()
        else:
            merged_opacity = opacities.mean(dim=0)
        
        return {
            'mean': merged_mean,
            'scale': merged_scale,
            'rotation': merged_rotation,
            'opacity': merged_opacity,
            'semantic': merged_sem,
            'uncertainty': merged_unc,
            'velocity': velocities.mean(dim=0) if velocities is not None else None,
            'evidential_alpha': (
                1.0 + (evidential_alphas - 1.0).clamp_min(0.0).sum(dim=0)
                if evidential_alphas is not None else None
            ),
        }
    
    def _apply_mask(self, gaussians, mask):
        """Apply mask to GaussianPrediction object"""
        B, G = mask.shape
        
        pruned_batches = []
        for b in range(B):
            batch_mask = mask[b]
            pruned_batches.append(GaussianPrediction(
                means=gaussians.means[b][batch_mask],
                scales=gaussians.scales[b][batch_mask],
                rotations=gaussians.rotations[b][batch_mask],
                opacities=gaussians.opacities[b][batch_mask],
                semantics=gaussians.semantics[b][batch_mask],
                uncertainties=gaussians.uncertainties[b][batch_mask] if gaussians.uncertainties is not None else None,
                evidential_alphas=gaussians.evidential_alphas[b][batch_mask] if gaussians.evidential_alphas is not None else None,
                original_means=gaussians.original_means[b][batch_mask] if gaussians.original_means is not None else None,
                delta_means=gaussians.delta_means[b][batch_mask] if gaussians.delta_means is not None else None,
                velocities=gaussians.velocities[b][batch_mask] if gaussians.velocities is not None else None,
            ))
        
        # Find max size and pad
        max_G = max([pb.means.shape[0] for pb in pruned_batches])
        device = pruned_batches[0].means.device
        dtype = pruned_batches[0].means.dtype
        
        padded_means = []
        padded_scales = []
        padded_rotations = []
        padded_opacities = []
        padded_semantics = []
        padded_uncertainties = []
        padded_alphas = []
        padded_original_means = []
        padded_delta_means = []
        padded_velocities = []
        
        for pb in pruned_batches:
            G_pruned = pb.means.shape[0]
            pad_size = max_G - G_pruned
            
            padded_means.append(torch.cat([pb.means, torch.zeros(pad_size, 3, device=device, dtype=dtype)], dim=0))
            padded_scales.append(torch.cat([pb.scales, torch.zeros(pad_size, 3, device=device, dtype=dtype)], dim=0))
            padded_rotations.append(torch.cat([pb.rotations, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(pad_size, 1)], dim=0))
            
            if len(pb.opacities.shape) == 1:
                padded_opacities.append(torch.cat([pb.opacities, torch.zeros(pad_size, device=device, dtype=dtype)], dim=0))
            else:
                padded_opacities.append(torch.cat([pb.opacities, torch.zeros(pad_size, pb.opacities.shape[1], device=device, dtype=dtype)], dim=0))
            
            padded_semantics.append(torch.cat([pb.semantics, torch.zeros(pad_size, pb.semantics.shape[1], device=device, dtype=dtype)], dim=0))
            
            if pb.uncertainties is not None:
                padded_uncertainties.append(torch.cat([pb.uncertainties, torch.ones(pad_size, device=device, dtype=dtype)], dim=0))
            else:
                padded_uncertainties.append(None)
            
            if pb.evidential_alphas is not None:
                padded_alphas.append(torch.cat([pb.evidential_alphas, torch.zeros(pad_size, pb.evidential_alphas.shape[1], device=device, dtype=dtype)], dim=0))
            else:
                padded_alphas.append(None)
            
            if pb.original_means is not None:
                padded_original_means.append(torch.cat([pb.original_means, torch.zeros(pad_size, 3, device=device, dtype=dtype)], dim=0))
            else:
                padded_original_means.append(None)
            
            if pb.delta_means is not None:
                padded_delta_means.append(torch.cat([pb.delta_means, torch.zeros(pad_size, 3, device=device, dtype=dtype)], dim=0))
            else:
                padded_delta_means.append(None)
            if pb.velocities is not None:
                padded_velocities.append(torch.cat([
                    pb.velocities, torch.zeros(pad_size, 3, device=device, dtype=dtype)
                ], dim=0))
            else:
                padded_velocities.append(None)
        
        return GaussianPrediction(
            means=torch.stack(padded_means, dim=0),
            scales=torch.stack(padded_scales, dim=0),
            rotations=torch.stack(padded_rotations, dim=0),
            opacities=torch.stack(padded_opacities, dim=0),
            semantics=torch.stack(padded_semantics, dim=0),
            uncertainties=torch.stack(padded_uncertainties, dim=0) if padded_uncertainties[0] is not None else None,
            evidential_alphas=torch.stack(padded_alphas, dim=0) if padded_alphas[0] is not None else None,
            original_means=torch.stack(padded_original_means, dim=0) if padded_original_means[0] is not None else None,
            delta_means=torch.stack(padded_delta_means, dim=0) if padded_delta_means[0] is not None else None,
            velocities=torch.stack(padded_velocities, dim=0) if padded_velocities[0] is not None else None,
        )
    
    def forward(self, gaussians, sensor_positions):
        """
        Complete refinement pipeline: Pruning → Merging
        
        Args:
            gaussians: GaussianPrediction object (must be batched, must contain uncertainties)
            sensor_positions: [B, G, 3] or [B, 3] - Sensor position(s) for pruning (batched)
                - [B, G, 3]: Each Gaussian has its own sensor position (recommended)
                  This should be the sensor position when each Gaussian was detected.
                - [B, 3]: Single sensor position for all Gaussians (fallback)
        
        Returns:
            refined_gaussians: Refined GaussianPrediction object (batched)
        """
        if gaussians is None:
            raise ValueError("gaussians cannot be None")
        if sensor_positions is None:
            raise ValueError("sensor_positions cannot be None for pruning")
        
        if self.verbose:
            B, initial_num, _ = gaussians.means.shape
            print(f"[GaussianRefinement] Starting refinement with {initial_num} Gaussians (batch size: {B})")
        
        # Step 1: Pruning (relative pruning strategy)
        pruned_gaussians, _ = self.prune_gaussians(gaussians, sensor_positions)
        
        # Step 2: Merging (semantic consistency check + MoM/DST merging)
        merged_gaussians = self.merge_gaussians(pruned_gaussians)
        
        if self.verbose:
            B, final_num, _ = merged_gaussians.means.shape
            print(f"[GaussianRefinement] Refinement complete: {initial_num} → {final_num} Gaussians (batch size: {B})")
        
        return merged_gaussians
