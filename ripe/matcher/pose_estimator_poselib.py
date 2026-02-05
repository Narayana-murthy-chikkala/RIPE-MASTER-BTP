import torch
from typing import Optional, Tuple


class PoseLibRelativePoseEstimator:
    """
    Deterministic fundamental matrix estimator

    - No RANSAC
    - No randomness
    - IRLS robust refinement
    - Same inliers every run
    """

    def __init__(self, iterations: int = 10):
        self.iterations = iterations

    def _normalize_points(self, pts):
        mean = pts.mean(0)
        std = pts.std(0) + 1e-8

        T = torch.tensor([
            [1/std[0], 0, -mean[0]/std[0]],
            [0, 1/std[1], -mean[1]/std[1]],
            [0, 0, 1]
        ], device=pts.device)

        pts_h = torch.cat([pts, torch.ones(len(pts),1, device=pts.device)], dim=1)
        pts_norm = (T @ pts_h.T).T

        return pts_norm[:, :2], T

    def _eight_point(self, pts1, pts2, weights=None):

        x1, y1 = pts1[:,0], pts1[:,1]
        x2, y2 = pts2[:,0], pts2[:,1]

        A = torch.stack([
            x1*x2, x1*y2, x1,
            y1*x2, y1*y2, y1,
            x2, y2, torch.ones_like(x1)
        ], dim=1)

        if weights is not None:
            A = A * weights[:,None]

        _, _, V = torch.linalg.svd(A)
        F = V[-1].view(3,3)

        U,S,V = torch.linalg.svd(F)
        S[-1] = 0
        F = U @ torch.diag(S) @ V

        return F

    def __call__(
        self,
        pts0: torch.Tensor,
        pts1: torch.Tensor,
        inl_th: float,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:

        if pts0.shape[0] < 8:
            return None, torch.zeros(len(pts0), dtype=torch.bool)

        pts0n, T0 = self._normalize_points(pts0)
        pts1n, T1 = self._normalize_points(pts1)

        weights = torch.ones(len(pts0), device=pts0.device)

        for _ in range(self.iterations):

            F = self._eight_point(pts0n, pts1n, weights)

            pts0h = torch.cat([pts0n, torch.ones(len(pts0n),1, device=pts0.device)], dim=1)
            pts1h = torch.cat([pts1n, torch.ones(len(pts1n),1, device=pts0.device)], dim=1)

            errs = torch.abs(torch.sum(pts1h * (F @ pts0h.T).T, dim=1))

            weights = 1 / (errs + 1e-6)

        F = T1.T @ F @ T0

        inliers = errs < inl_th

        return F, inliers
