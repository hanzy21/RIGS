"""
E2-BKI Main Inference Module

This module provides a unified interface for E2-BKI inference, integrating
Gaussian Refinement (optional) and Evidential Ellipsoidal BKI.

Reference:
    Kim et al., "E2-BKI: Evidential Ellipsoidal Bayesian Kernel Inference", 2025

Note:
    - Gaussian Refinement interface is preserved but not called by default
    - Set refinement_enabled=True in config to enable refinement
    - All inputs are assumed to be batched (batch dimension always present)
"""

import torch
import torch.nn as nn
from ..encoder.gaussian_encoder.utils import GaussianPrediction
from .anisotropic_kernel import AnisotropicKernel
from .evidential_bki import EvidentialEllipsoidalBKI


class E2BKIInference(nn.Module):
    """
    E2-BKI Main Inference Module
    
    This module provides a unified interface for E2-BKI inference, integrating:
    1. Gaussian Refinement (optional, controlled by refinement_enabled)
    2. Evidential Ellipsoidal BKI (always used)
    
    Key Features:
        - Preserves GaussianRefinement interface but doesn't call it by default
        - Provides unified forward interface for E2-BKI inference
        - Handles configuration and component initialization
    
    Args:
        config (dict): E2-BKI configuration dictionary
            - kernel_scale (float): Kernel scale ℓ in meters (default: 0.2)
            - refinement_enabled (bool): Whether to enable Gaussian refinement (default: False)
            - use_uncertainty_decomposition (bool): Whether to decompose uncertainty (default: True)
            - beta (float): Uncertainty sensitivity β (default: 0.75)
            - u_percentile (float): Uncertainty threshold percentile ũ (default: 0.1)
            - use_uncertainty_adaptive_kernel (bool): Use uncertainty-adaptive kernel (default: True)
            - use_ellipsoid_distance (bool): Use ellipsoid surface distance (default: False)
            - tau (float): Ellipsoid size parameter τ (default: 1.0)
            - alpha_0 (float): Dirichlet prior α_0^c (default: 0.001)
            - use_temporal_recursion (bool): Whether to use temporal recursion (default: True)
            - dL (float): Large radius for refinement (default: 5 * kernel_scale)
            - dS (float): Small radius for refinement (default: kernel_scale)
            - epsilon (float): Pruning sensitivity parameter ϵ (default: 2.5)
            - sparsity_weight (float): Weight for sparsity uncertainty (default: 1.0)
            - verbose (bool): Whether to print debug information (default: False)
            - device (str): Device for computation (default: 'cuda')
            - global_voxel_size (float): Voxel size for global coordinate quantization in key generation.
                                        Allows tolerance for quantization errors in local-to-global transformation.
                                        Recommended: 0.2m (half of local voxel_size) or larger.
                                        If None, uses 1mm precision (default: None)
            - normalize_evidence_by_kernel (bool): Normalize evidence by kernel mass (default: False)
    """
    
    def __init__(self, config):
        """
        Initialize E2-BKI Inference Module
        
        Args:
            config (dict): E2-BKI configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Extract configuration parameters with defaults
        kernel_scale = config.get('kernel_scale', 0.2)
        refinement_enabled = config.get('refinement_enabled', False)  # Default: False
        use_uncertainty_decomposition = config.get('use_uncertainty_decomposition', True)
        beta = config.get('beta', 0.75)
        u_percentile = config.get('u_percentile', 0.1)
        use_uncertainty_adaptive_kernel = config.get('use_uncertainty_adaptive_kernel', True)
        use_ellipsoid_distance = config.get('use_ellipsoid_distance', False)
        tau = config.get('tau', 1.0)
        alpha_0 = config.get('alpha_0', 0.001)
        use_temporal_recursion = config.get('use_temporal_recursion', True)
        sparsity_weight = config.get('sparsity_weight', 1.0)
        verbose = config.get('verbose', False)
        device = config.get('device', 'cuda')
        global_voxel_size = config.get('global_voxel_size', None)  # For tolerant key generation
        
        # Initialize AnisotropicKernel
        # Use paper's k' function if uncertainty-adaptive kernel is enabled
        use_paper_kernel = use_uncertainty_adaptive_kernel
        # Distance filtering parameter
        max_euclidean_distance = config.get('max_euclidean_distance', 5.0)  # Default: 5.0m
        # Blockwise pruning parameter
        use_blockwise_pruning = config.get('use_blockwise_pruning', True)  # Default: True
        block_size = config.get('block_size', 8192)  # Default: 8192
        blockwise_preselect_gaussians = config.get('blockwise_preselect_gaussians', 128)
        
        self.kernel = AnisotropicKernel(
            scale=kernel_scale,
            eps=1e-8,
            max_mahalanobis_dist=50.0,
            verbose=verbose,
            use_paper_kernel=use_paper_kernel,
            max_euclidean_distance=max_euclidean_distance,  # Only compute kernel for Gaussians within this distance
            use_blockwise_pruning=use_blockwise_pruning,  # Enable/disable blockwise pruning
            block_size=block_size,  # Block size for blockwise pruning
            blockwise_preselect_gaussians=blockwise_preselect_gaussians
        )
        
        # Initialize EvidentialEllipsoidalBKI
        # Sparse kernel optimization parameters
        use_sparse_kernel = config.get('use_sparse_kernel', True)  # Default: True
        kernel_threshold = config.get('kernel_threshold', 1e-6)  # Default: 1e-6
        max_kernels_per_point = config.get('max_kernels_per_point', 10)  # Default: 10
        
        # Extract evidence_scale from evidential_head_config if available
        evidential_head_config = config.get('evidential_head_config', {})
        evidence_scale = evidential_head_config.get('evidence_scale', 1.0)
        # Extract evidence_weights from config (not from evidential_head_config)
        evidence_weights = config.get('evidence_weights', None)  # Per-class evidence weights
        # Extract alpha_m_cap from config (evidence saturation cap)
        alpha_m_cap = config.get('alpha_m_cap', None)  # Per-class alpha_m upper limit
        temporal_alpha_decay = config.get('temporal_alpha_decay', 1.0)  # Alpha_prev decay (1.0 = no decay)
        temporal_alpha_write_aggregation = config.get('temporal_alpha_write_aggregation', 'last_writer')  # Scheme B: 'last_writer' | 'voxel_mean'
        temporal_require_frame_id = config.get('temporal_require_frame_id', True)
        timing_silent = config.get('e2bki_timing_silent', False)  # Suppress timing print (for forward_profile)
        enable_timing_profile = config.get('enable_timing_profile', False)
        
        if config.get('semantic_input_mode', 'evidential') != 'evidential':
            raise ValueError("RIGS requires evidential two-channel Gaussian semantics")
        if config.get('evidence_aggregation_mode', 'standard') != 'standard':
            raise ValueError("RIGS uses standard evidence aggregation")
        # Kernel normalization: divide evidence by kernel mass to be density-invariant
        normalize_evidence_by_kernel = config.get('normalize_evidence_by_kernel', False)
        
        self.bki = EvidentialEllipsoidalBKI(
            kernel=self.kernel,
            use_uncertainty_decomposition=use_uncertainty_decomposition,
            sparsity_weight=sparsity_weight,
            eps=1e-8,
            verbose=verbose,
            beta=beta,
            u_percentile=u_percentile,
            use_uncertainty_adaptive_kernel=use_uncertainty_adaptive_kernel,
            use_ellipsoid_distance=use_ellipsoid_distance,
            tau=tau,
            alpha_0=alpha_0,
            use_temporal_recursion=use_temporal_recursion,
            # Sparse kernel optimization parameters
            use_sparse_kernel=use_sparse_kernel,
            kernel_threshold=kernel_threshold,
            max_kernels_per_point=max_kernels_per_point,
            # Evidence scale for amplifying evidence increment
            evidence_scale=evidence_scale,
            # Per-class evidence weights (e.g., to reduce empty class evidence)
            evidence_weights=evidence_weights,
            # Per-class alpha_m saturation cap (upper limit)
            alpha_m_cap=alpha_m_cap,
            temporal_alpha_decay=temporal_alpha_decay,
            temporal_alpha_write_aggregation=temporal_alpha_write_aggregation,
            temporal_require_frame_id=temporal_require_frame_id,
            timing_silent=timing_silent,
            enable_timing_profile=enable_timing_profile,
            normalize_evidence_by_kernel=normalize_evidence_by_kernel,
            # Aggregation mode parameters
            aggregation_mode=config.get('aggregation_mode', 'standard'),
            cc_norm_min_count=config.get('cc_norm_min_count', 1),
            attention_tau=config.get('attention_tau', 0.3),
        )
        
        # Set global_voxel_size for tolerant key generation TODO：check是否可行
        if global_voxel_size is not None:
            self.bki.global_voxel_size = global_voxel_size
        
        # Initialize GaussianRefinement (preserve interface but don't call by default)
        # Only initialize if refinement_enabled is True
        if refinement_enabled:
            from .gaussian_refinement import GaussianRefinement
            dL = config.get('dL', None)  # Default: 5 * kernel_scale
            dS = config.get('dS', None)  # Default: kernel_scale
            epsilon = config.get('epsilon', 2.5)
            self.refinement = GaussianRefinement(
                kernel_scale=kernel_scale,
                dL=dL,
                dS=dS,
                epsilon=epsilon,
                device=device,
                verbose=verbose
            )
        else:
            self.refinement = None
        
        self.refinement_enabled = refinement_enabled
        self.verbose = verbose
        
    
    def forward(self, gaussians, query_points, sensor_positions=None, query_points_global=None, frame_id=None, base_probs=None):
        """
        E2-BKI inference main function
        
        Complete inference pipeline:
        1. (Optional) Gaussian Refinement if enabled and sensor_positions provided
        2. Evidential Ellipsoidal BKI inference
        
        Args:
            gaussians: GaussianPrediction object (must be batched)
                - means: [B, G, 3] - Gaussian centers in local (ego) coordinate system
                - scales: [B, G, 3] - Gaussian scales
                - rotations: [B, G, 4] - Rotation quaternions
                - semantics: [B, G, C] - Semantic probabilities
                - uncertainties: [B, G] - Semantic uncertainties (optional)
            query_points: [B, N, 3] - Query point coordinates in local (ego) coordinate system (batched)
                Used for kernel calculation (must be in same coordinate system as gaussians.means)
            sensor_positions: [B, G, 3] or [B, 3] or None - Sensor positions (optional)
                - Only used if refinement is enabled
                - If None and refinement is enabled, refinement will be skipped
            query_points_global: [B, N, 3] or None - Query point coordinates in global coordinate system (batched)
                Used for temporal recursion to generate consistent keys across frames
                If None, uses query_points as fallback (not recommended for temporal recursion)
            frame_id: Optional sequential frame identifier for temporal recursion.
        
        Returns:
            result: dict containing
                - semantic_map: [B, N, C] - Semantic probability predictions E[θ̂^c_m]
                - uncertainty_map: [B, N] - Total uncertainty values u_m,total
                - alpha_m: [B, N, C] - Updated Dirichlet parameters α^c_m
                - sparsity_uncertainty: [B, N] - Sparsity uncertainty u_m,spa (if decomposition enabled)
                - semantic_uncertainty: [B, N] - Semantic uncertainty u_m,sem (if decomposition enabled)
        """
        # Input validation
        if gaussians is None:
            raise ValueError("gaussians cannot be None")
        if query_points is None:
            raise ValueError("query_points cannot be None")
        
        # Validate input shapes (must be batched)
        if len(query_points.shape) != 3 or query_points.shape[2] != 3:
            raise ValueError(f"query_points must be [B, N, 3], got {query_points.shape}")
        
        if not hasattr(gaussians, 'means'):
            raise ValueError("gaussians must be a GaussianPrediction object with 'means' attribute")
        
        if len(gaussians.means.shape) != 3 or gaussians.means.shape[2] != 3:
            raise ValueError(f"gaussians.means must be [B, G, 3], got {gaussians.means.shape}")
        
        # Step 1: (Optional) Gaussian Refinement
        # Only call refinement if:
        #   1. refinement is initialized (refinement_enabled=True)
        #   2. sensor_positions is provided
        if self.refinement is not None and sensor_positions is not None:
            if self.verbose:
                print("[E2BKIInference] Applying Gaussian Refinement...")
            refined_gaussians = self.refinement.forward(gaussians, sensor_positions)
        else:
            # Default: skip refinement, use original gaussians
            if self.verbose and self.refinement is not None and sensor_positions is None:
                print("[E2BKIInference] Refinement enabled but sensor_positions not provided, skipping refinement")
            refined_gaussians = gaussians

        # Step 1.5: Convert native evidential semantics to BKI probabilities.
        try:
            sem = getattr(refined_gaussians, 'semantics', None)
            if sem is None:
                raise ValueError("refined_gaussians.semantics is None")
            if len(sem.shape) != 3:
                raise ValueError(f"refined_gaussians.semantics must be [B,G,C], got {sem.shape}")

            target_c = sem.shape[-1]
            if target_c != 2:
                raise ValueError(f"RIGS BKI requires two non-empty semantics, got {sem.shape}")

            ev_alphas = getattr(refined_gaussians, 'evidential_alphas', None)
            if ev_alphas is None or ev_alphas.dim() != 3 or ev_alphas.shape[-1] != 2:
                raise ValueError(
                    "evidential inference requires alpha with shape [B,G,2], got "
                    f"{getattr(ev_alphas, 'shape', None)}"
                )
            sem = ev_alphas / (ev_alphas.sum(dim=-1, keepdim=True) + 1e-8)
            sem = torch.clamp(sem, min=0.0, max=1.0)

            # Update GaussianPrediction semantics
            if hasattr(refined_gaussians, "_replace"):
                refined_gaussians = refined_gaussians._replace(semantics=sem)
            else:
                from model.encoder.gaussian_encoder.utils import GaussianPrediction
                refined_gaussians = GaussianPrediction(
                    means=refined_gaussians.means,
                    scales=refined_gaussians.scales,
                    rotations=refined_gaussians.rotations,
                    opacities=getattr(refined_gaussians, 'opacities', None),
                    semantics=sem,
                    uncertainties=getattr(refined_gaussians, 'uncertainties', None),
                    evidential_alphas=getattr(refined_gaussians, 'evidential_alphas', None),
                    original_means=getattr(refined_gaussians, 'original_means', None),
                    delta_means=getattr(refined_gaussians, 'delta_means', None),
                )
        except Exception:
            raise
        
        # Step 2: Evidential Ellipsoidal BKI inference
        if self.verbose:
            print("[E2BKIInference] Running Evidential Ellipsoidal BKI inference...")
        
        # Pass global query points and the frame identifier for temporal recursion.
        bki_result = self.bki.forward(query_points, refined_gaussians, query_points_global=query_points_global, frame_id=frame_id, base_probs=base_probs)
        
        # Rename keys for consistency with expected output format
        result = {
            'semantic_map': bki_result['semantic_probs'],  # [B, N, C]
            'uncertainty_map': bki_result['uncertainties'],  # [B, N]
            'alpha_m': bki_result['alpha_m'],  # [B, N, C]
        }
        
        # Add uncertainty decomposition components if available
        if 'sparsity_uncertainty' in bki_result:
            result['sparsity_uncertainty'] = bki_result['sparsity_uncertainty']  # [B, N]
        if 'semantic_uncertainty' in bki_result:
            result['semantic_uncertainty'] = bki_result['semantic_uncertainty']  # [B, N]
        if result['semantic_map'].shape[-1] != 2 or result['alpha_m'].shape[-1] != 2:
            raise ValueError("native RIGS BKI output must have two non-empty channels")
        
        if self.verbose:
            B, N, C = result['semantic_map'].shape
            print(f"[E2BKIInference] Inference complete: {B} batches, {N} query points, {C} classes")
        
        return result
    
    def __call__(self, gaussians, query_points, sensor_positions=None, query_points_global=None, frame_id=None, base_probs=None):
        """
        Support function call syntax
        
        Args:
            gaussians: GaussianPrediction object
            query_points: [B, N, 3] - Query point coordinates in local (ego) coordinate system
            sensor_positions: [B, G, 3] or [B, 3] or None - Sensor positions (optional)
            query_points_global: [B, N, 3] or None - Query point coordinates in global coordinate system (optional)
            frame_id: Optional sequential frame identifier for temporal recursion
            base_probs: [B, N, C] or None - Base model class probabilities for attention aggregation (optional)
        
        Returns:
            result: dict containing inference results
        """
        return self.forward(gaussians, query_points, sensor_positions, query_points_global, frame_id, base_probs=base_probs)
    
    def reset_alpha_history(self):
        """
        Reset alpha history for temporal recursion (per-point key).
        Clears both alpha_history and key_to_frame_id.
        """
        if hasattr(self.bki, 'alpha_history'):
            self.bki.alpha_history = {}
        if hasattr(self.bki, 'key_to_frame_id'):
            self.bki.key_to_frame_id = {}
        if self.verbose:
            print("[E2BKIInference] Alpha history reset")
