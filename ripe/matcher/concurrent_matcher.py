import torch
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ConcurrentMatcher:
    """Concurrent descriptor matcher with PROSAC-based robust estimation.
    
    This matcher handles batch processing of image pairs with descriptor matching,
    spatial filtering, and PROSAC-based geometric verification.
    
    Args:
        matcher (Callable): Descriptor matcher function (e.g., MNN matcher)
        robust_estimator (Callable): PROSAC-based robust estimator
        min_num_matches (int): Minimum matches required for PROSAC estimation
        max_matches (int): Maximum matches to keep after descriptor matching
        max_keypoints (int): Maximum keypoints to use per image
        spatial_th (float): Spatial distance threshold for filtering
    """

    def __init__(
        self,
        matcher: Callable,
        robust_estimator: Callable,
        min_num_matches: int = 8,
        max_matches: int = 200,
        max_keypoints: int = 1024,
        spatial_th: float = 300.0,
    ):
        self.matcher = matcher
        self.robust_estimator = robust_estimator
        self.min_num_matches = min_num_matches
        self.max_matches = max_matches
        self.max_keypoints = max_keypoints
        self.spatial_th = spatial_th

    def _cap_keypoints(self, desc, mask):
        """Limit keypoints to max_keypoints based on mask."""
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        idx = idx[: self.max_keypoints]
        return desc[idx], idx

    def _topk_matches(self, dists, idx_matches):
        """Keep only top-k best matches by distance."""
        k = min(self.max_matches, dists.shape[0])
        order = torch.argsort(dists.squeeze())
        best = order[:k]
        return dists[best], idx_matches[best]

    def _spatial_filter(self, kpts1, kpts2, idx_matches):
        """Filter matches by spatial distance threshold."""
        pts1 = kpts1[idx_matches[:, 0]]
        pts2 = kpts2[idx_matches[:, 1]]
        dist = torch.norm(pts1 - pts2, dim=1)
        keep = dist < self.spatial_th
        return idx_matches[keep], keep

    @torch.no_grad()
    def __call__(
        self,
        kpts1,
        kpts2,
        pdesc1,
        pdesc2,
        selected_mask1,
        selected_mask2,
        inl_th,
        label: Optional[torch.Tensor] = None,
    ):
        """
        Match descriptors and filter with PROSAC robust estimation.
        
        Args:
            kpts1 (torch.Tensor): Keypoints in image 1 (B, N1, 2)
            kpts2 (torch.Tensor): Keypoints in image 2 (B, N2, 2)
            pdesc1 (torch.Tensor): Descriptors in image 1 (B, N1, D)
            pdesc2 (torch.Tensor): Descriptors in image 2 (B, N2, D)
            selected_mask1 (torch.Tensor): Mask for valid keypoints in image 1 (B, N1)
            selected_mask2 (torch.Tensor): Mask for valid keypoints in image 2 (B, N2)
            inl_th (float): Inlier threshold for PROSAC
            label (torch.Tensor, optional): Ground truth labels for supervision
            
        Returns:
            Tuple containing:
                - batch_rel_idx_matches: Relative indices within selected keypoints
                - batch_idx_matches: Absolute indices into full keypoint arrays
                - batch_prosac_inliers: Boolean masks of PROSAC inliers
                - batch_Fm: Fundamental matrices from PROSAC estimation
        """

        device = pdesc1.device
        batch_size = pdesc1.shape[0]

        batch_rel_idx_matches = [None] * batch_size
        batch_idx_matches = [None] * batch_size
        batch_prosac_inliers = [None] * batch_size
        batch_Fm = [None] * batch_size

        for b in range(batch_size):

            # Cap keypoints based on selection mask
            desc1, idx1 = self._cap_keypoints(pdesc1[b], selected_mask1[b])
            desc2, idx2 = self._cap_keypoints(pdesc2[b], selected_mask2[b])

            # Skip if not enough keypoints
            if desc1.shape[0] < 16 or desc2.shape[0] < 16:
                continue

            # Match descriptors
            dists, idx_matches = self.matcher(desc1, desc2)
            dists, idx_matches = self._topk_matches(dists, idx_matches)

            # Apply spatial filtering
            idx_matches, keep_mask = self._spatial_filter(
                kpts1[b][idx1],
                kpts2[b][idx2],
                idx_matches
            )

            if idx_matches.shape[0] == 0:
                continue

            # Create ranking from match distances (better matches have higher rank)
            ranking = -dists.squeeze()[keep_mask]

            batch_rel_idx_matches[b] = idx_matches.clone()

            # Convert to absolute indices
            idx_matches[:, 0] = idx1[idx_matches[:, 0]]
            idx_matches[:, 1] = idx2[idx_matches[:, 1]]
            batch_idx_matches[b] = idx_matches

            num_matches = idx_matches.shape[0]

            # If not enough matches for PROSAC, mark all as outliers
            if num_matches < self.min_num_matches:
                batch_prosac_inliers[b] = torch.zeros(
                    num_matches, device=device, dtype=torch.bool
                )
                continue

            # Extract matched keypoints
            mkpts1 = kpts1[b][idx_matches[:, 0]]
            mkpts2 = kpts2[b][idx_matches[:, 1]]

            # Run PROSAC robust estimation with ranking
            Fm, inl = self.robust_estimator(mkpts1, mkpts2, inl_th, ranking)

            batch_prosac_inliers[b] = inl
            batch_Fm[b] = Fm

        return (
            batch_rel_idx_matches,
            batch_idx_matches,
            batch_prosac_inliers,
            batch_Fm,
        )