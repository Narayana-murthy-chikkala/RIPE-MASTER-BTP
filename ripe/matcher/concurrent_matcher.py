import torch
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ConcurrentMatcher:
    """
    Deterministic real-time matcher

    Improvements:
    - Keep only strongest matches (quality > quantity)
    - Spatial sanity filtering
    - PROSAC-style ordering (best first)
    - Stable deterministic behavior
    """

    def __init__(
        self,
        matcher: Callable,
        robust_estimator: Callable,
        min_num_matches: int = 8,
        max_matches: int = 200,
        max_keypoints: int = 1024,
        spatial_th: float = 300.0,   # 🔥 NEW: position sanity threshold (pixels)
    ):
        self.matcher = matcher
        self.robust_estimator = robust_estimator
        self.min_num_matches = min_num_matches
        self.max_matches = max_matches
        self.max_keypoints = max_keypoints
        self.spatial_th = spatial_th

    def _cap_keypoints(self, desc, mask):
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        idx = idx[: self.max_keypoints]
        return desc[idx], idx

    def _topk_matches(self, dists, idx_matches):
        """
        Keep only strongest matches.
        Smaller distance = stronger match.
        """

        k = min(self.max_matches, dists.shape[0])

        # sort by distance (best first)
        order = torch.argsort(dists.squeeze())
        best = order[:k]

        return dists[best], idx_matches[best]

    def _spatial_filter(self, kpts1, kpts2, idx_matches):
        """
        Remove matches that jump unrealistically far.
        Basic sanity check.
        """

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

        device = pdesc1.device
        batch_size = pdesc1.shape[0]

        batch_rel_idx_matches = [None] * batch_size
        batch_idx_matches = [None] * batch_size
        batch_ransac_inliers = [None] * batch_size
        batch_Fm = [None] * batch_size

        for b in range(batch_size):

            desc1, idx1 = self._cap_keypoints(pdesc1[b], selected_mask1[b])
            desc2, idx2 = self._cap_keypoints(pdesc2[b], selected_mask2[b])

            if desc1.shape[0] < 16 or desc2.shape[0] < 16:
                continue

            # MNN matching
            dists, idx_matches = self.matcher(desc1, desc2)

            # keep only strongest matches
            dists, idx_matches = self._topk_matches(dists, idx_matches)

            # spatial sanity filter
            idx_matches, keep_mask = self._spatial_filter(
                kpts1[b][idx1],
                kpts2[b][idx2],
                idx_matches
            )

            if idx_matches.shape[0] == 0:
                continue

            batch_rel_idx_matches[b] = idx_matches.clone()

            idx_matches[:, 0] = idx1[idx_matches[:, 0]]
            idx_matches[:, 1] = idx2[idx_matches[:, 1]]
            batch_idx_matches[b] = idx_matches

            num_matches = idx_matches.shape[0]

            if num_matches < self.min_num_matches:
                batch_ransac_inliers[b] = torch.zeros(
                    num_matches, device=device, dtype=torch.bool
                )
                continue

            if label is not None and label[b] == 0:
                batch_ransac_inliers[b] = torch.ones(
                    num_matches, device=device, dtype=torch.bool
                )
                continue

            mkpts1 = kpts1[b][idx_matches[:, 0]]
            mkpts2 = kpts2[b][idx_matches[:, 1]]

            # PROSAC behavior = already sorted by strength
            Fm, inl = self.robust_estimator(mkpts1, mkpts2, inl_th)

            batch_ransac_inliers[b] = inl
            batch_Fm[b] = Fm

        return (
            batch_rel_idx_matches,
            batch_idx_matches,
            batch_ransac_inliers,
            batch_Fm,
        )
