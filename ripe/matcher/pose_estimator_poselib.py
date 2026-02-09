import torch
import random
from math import comb


class PROSACRelativePoseEstimator:
    """
    True PROSAC (Progressive Sample Consensus) fundamental matrix estimator.
    
    PROSAC is a robust estimation algorithm that combines sampling with prioritization.
    Unlike RANSAC which samples uniformly, PROSAC progressively examines samples
    starting with the most promising correspondences (ranked by match quality).
    
    Features:
    - Ranked correspondences: Uses match distances to rank correspondences
    - Official PROSAC sampling schedule: Implements the theoretical growth schedule
    - Sampson error scoring: Uses Sampson distance for geometric validation
    - Deterministic: Reproducible results with fixed seed
    - IRLS refinement: Iteratively reweighted least squares for better estimation
    - Point normalization: Normalizes points for numerical stability

    Args:
        iterations (int): Maximum PROSAC iterations (default: 5000)
        irls_iters (int): Number of IRLS refinement iterations (default: 10)
        seed (int): Random seed for reproducibility (default: 32000)
    """

    def __init__(self, iterations=5000, irls_iters=10, seed=32000):
        self.iterations = iterations
        self.irls_iters = irls_iters
        self.seed = seed
        self.m = 8  # Minimum matches for fundamental matrix (8-point algorithm)

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _normalize_points(self, pts):
        """Normalize points to improve numerical stability.
        
        Applies similarity transformation to center points at origin and scale
        to have unit standard deviation. This is critical for robust estimation.
        
        Args:
            pts (torch.Tensor): Points of shape (N, 2)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Normalized 2D points (N, 2)
                - Normalization transformation matrix (3, 3) for denormalization
        """
        mean = pts.mean(0)
        std = pts.std(0) + 1e-8

        T = torch.tensor([
            [1/std[0], 0, -mean[0]/std[0]],
            [0, 1/std[1], -mean[1]/std[1]],
            [0, 0, 1]
        ], device=pts.device, dtype=pts.dtype)

        pts_h = torch.cat([pts, torch.ones(len(pts), 1, device=pts.device, dtype=pts.dtype)], dim=1)
        pts_norm = (T @ pts_h.T).T
        return pts_norm[:, :2], T

    def _eight_point(self, pts1, pts2, weights=None):
        """Compute fundamental matrix using 8-point algorithm.
        
        Solves the linear system Af = 0 where f is vectorized fundamental matrix.
        Uses SVD to find least-squares solution. Enforces rank-2 constraint by
        zeroing smallest singular value.
        
        Args:
            pts1 (torch.Tensor): Points in image 1 (N, 2)
            pts2 (torch.Tensor): Points in image 2 (N, 2)
            weights (torch.Tensor, optional): Sample weights for IRLS
            
        Returns:
            torch.Tensor: Fundamental matrix (3, 3)
        """
        x1, y1 = pts1[:, 0], pts1[:, 1]
        x2, y2 = pts2[:, 0], pts2[:, 1]

        # Build constraint matrix
        A = torch.stack([
            x1*x2, x1*y2, x1,
            y1*x2, y1*y2, y1,
            x2, y2, torch.ones_like(x1)
        ], dim=1)

        if weights is not None:
            A = A * weights[:, None]

        # Solve using SVD: F is the last column of V (right null space)
        _, _, V = torch.linalg.svd(A)
        F = V[-1].view(3, 3)

        # Enforce rank-2 constraint
        U, S, V = torch.linalg.svd(F)
        S[-1] = 0
        F = U @ torch.diag(S) @ V

        return F

    def _sampson_error(self, F, pts0h, pts1h):
        """Compute Sampson error for epipolar constraint.
        
        Sampson error is a geometric error that measures the distance between
        corresponding points and the epipolar line, accounting for both directions.
        
        Args:
            F (torch.Tensor): Fundamental matrix (3, 3)
            pts0h (torch.Tensor): Homogeneous points in image 1 (N, 3)
            pts1h (torch.Tensor): Homogeneous points in image 2 (N, 3)
            
        Returns:
            torch.Tensor: Sampson error for each correspondence (N,)
        """
        Fx1 = (F @ pts0h.T).T
        Ftx2 = (F.T @ pts1h.T).T
        denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
        err = torch.sum(pts1h * Fx1, dim=1)**2 / (denom + 1e-8)
        return err

    def _irls(self, pts0, pts1):
        """Iteratively Reweighted Least Squares refinement.
        
        Refines fundamental matrix by iteratively reweighting samples based on
        their residuals. Samples with larger errors get lower weights.
        
        Args:
            pts0 (torch.Tensor): Normalized points in image 1 (N, 2)
            pts1 (torch.Tensor): Normalized points in image 2 (N, 2)
            
        Returns:
            torch.Tensor: Refined fundamental matrix (3, 3)
        """
        weights = torch.ones(len(pts0), device=pts0.device, dtype=pts0.dtype)

        for _ in range(self.irls_iters):
            F = self._eight_point(pts0, pts1, weights)

            pts0h = torch.cat([pts0, torch.ones(len(pts0), 1, device=pts0.device, dtype=pts0.dtype)], dim=1)
            pts1h = torch.cat([pts1, torch.ones(len(pts1), 1, device=pts1.device, dtype=pts1.dtype)], dim=1)

            errs = self._sampson_error(F, pts0h, pts1h)
            weights = 1 / (errs + 1e-6)

        return F

    def _prosac_growth(self, N):
        """Generate PROSAC sampling schedule using theoretical growth function.
        
        PROSAC grows the sample set N gradually from m (minimum) to full set.
        At each step n, it samples (m-1) points from top n and always includes
        point n. The number of iterations at each step is determined by the
        binomial coefficient ratio.
        
        Args:
            N (int): Total number of correspondences
            
        Returns:
            List[int]: Schedule indicating sample set size at each iteration
        """
        T_N = comb(N, self.m)
        schedule = []

        for n in range(self.m, N + 1):
            T_n = comb(n, self.m)
            T_n1 = comb(n - 1, self.m) if n > self.m else 0
            growth = int(self.iterations * (T_n - T_n1) / T_N)
            schedule += [n] * max(growth, 1)

        return schedule[:self.iterations]

    def __call__(self, pts0, pts1, inl_th, ranking=None):
        """
        Estimate fundamental matrix using PROSAC with ranked correspondences.
        
        Progressive sampling strategy:
        1. Start with best-ranked correspondences
        2. Grow sample set according to PROSAC schedule
        3. Sample (m-1) points from current set, always include newest point
        4. Estimate fundamental matrix via IRLS
        5. Score on all points using Sampson error
        6. Keep best model
        
        Args:
            pts0 (torch.Tensor): Correspondences in image 1 (N, 2)
            pts1 (torch.Tensor): Correspondences in image 2 (N, 2)
            inl_th (float): Sampson error threshold for inlier classification
            ranking (torch.Tensor, optional): Ranking scores (higher = better)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Fundamental matrix (3, 3) or None if estimation fails
                - Boolean inlier mask (N,)
        """

        self._set_seed()

        if len(pts0) < self.m:
            return None, torch.zeros(len(pts0), dtype=torch.bool, device=pts0.device)

        N = len(pts0)

        # Sort by ranking if provided (higher ranking = better matches)
        if ranking is not None:
            idx = torch.argsort(ranking, descending=True)
            pts0 = pts0[idx]
            pts1 = pts1[idx]

        # Normalize points for numerical stability
        pts0n, T0 = self._normalize_points(pts0)
        pts1n, T1 = self._normalize_points(pts1)

        # Generate PROSAC sampling schedule
        schedule = self._prosac_growth(N)

        best_F = None
        best_inliers = None
        best_count = -1

        # Precompute homogeneous points for all correspondences
        pts0h_all = torch.cat([pts0n, torch.ones(N, 1, device=pts0.device, dtype=pts0.dtype)], dim=1)
        pts1h_all = torch.cat([pts1n, torch.ones(N, 1, device=pts1.device, dtype=pts1.dtype)], dim=1)

        # PROSAC iterations
        for n in schedule:
            # Sample m-1 points from top n-1, always include point n
            sample = random.sample(range(n - 1), self.m - 1)
            sample.append(n - 1)

            # Estimate fundamental matrix with IRLS refinement
            F = self._irls(pts0n[sample], pts1n[sample])

            # Score on all points
            errs = self._sampson_error(F, pts0h_all, pts1h_all)
            inliers = errs < inl_th
            count = inliers.sum().item()

            # Update best model
            if count > best_count:
                best_count = count
                best_F = F
                best_inliers = inliers

        # Denormalize fundamental matrix
        if best_F is not None:
            best_F = T1.T @ best_F @ T0

        return best_F, best_inliers