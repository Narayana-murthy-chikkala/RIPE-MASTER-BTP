import torch
from pose_estimator_poselib import PROSACRelativePoseEstimator


def test_repeatability(estimator, pts0, pts1, ranking, th=0.5, runs=5):
    """
    Test PROSAC stability by running multiple times and checking consistency.
    
    Args:
        estimator (PROSACRelativePoseEstimator): PROSAC estimator instance
        pts0 (torch.Tensor): Correspondences in image 1 (N, 2)
        pts1 (torch.Tensor): Correspondences in image 2 (N, 2)
        ranking (torch.Tensor): Ranking scores for correspondences
        th (float): Inlier threshold for PROSAC
        runs (int): Number of runs to test repeatability
    """
    counts = []

    for run_idx in range(runs):
        _, inl = estimator(pts0, pts1, th, ranking)
        count = inl.sum().item()
        counts.append(count)
        print(f"Run {run_idx + 1}: {count} inliers")

    print(f"\nInlier counts across {runs} runs: {counts}")
    print(f"Mean inliers: {sum(counts) / len(counts):.1f}")
    print(f"Std deviation: {torch.tensor(counts, dtype=torch.float).std():.2f}")

    if len(set(counts)) == 1:
        print("✅ Stable PROSAC - same result every run")
        return True
    else:
        print("⚠️ Variable results - different inlier counts across runs")
        print("   This may be expected due to random sampling in PROSAC")
        return False


def test_ranking_effect(estimator, pts0, pts1, th=0.5):
    """
    Test how ranking affects PROSAC results.
    
    Args:
        estimator (PROSACRelativePoseEstimator): PROSAC estimator instance
        pts0 (torch.Tensor): Correspondences in image 1 (N, 2)
        pts1 (torch.Tensor): Correspondences in image 2 (N, 2)
        th (float): Inlier threshold for PROSAC
    """
    print("\n" + "="*60)
    print("Testing Ranking Effect on PROSAC")
    print("="*60)
    
    N = len(pts0)
    
    # Test 1: With good ranking (based on match quality)
    good_ranking = torch.linspace(1.0, 0.0, N)
    print("\nTest 1: With good ranking (descending quality)")
    _, inl_good = estimator(pts0, pts1, th, ranking=good_ranking)
    print(f"Inliers with good ranking: {inl_good.sum().item()}/{N}")
    
    # Test 2: With random ranking
    random_ranking = torch.rand(N)
    print("\nTest 2: With random ranking")
    _, inl_random = estimator(pts0, pts1, th, ranking=random_ranking)
    print(f"Inliers with random ranking: {inl_random.sum().item()}/{N}")
    
    # Test 3: Without ranking (None)
    print("\nTest 3: Without ranking (None)")
    _, inl_none = estimator(pts0, pts1, th, ranking=None)
    print(f"Inliers without ranking: {inl_none.sum().item()}/{N}")


def test_threshold_sensitivity(estimator, pts0, pts1, ranking):
    """
    Test PROSAC sensitivity to inlier threshold.
    
    Args:
        estimator (PROSACRelativePoseEstimator): PROSAC estimator instance
        pts0 (torch.Tensor): Correspondences in image 1 (N, 2)
        pts1 (torch.Tensor): Correspondences in image 2 (N, 2)
        ranking (torch.Tensor): Ranking scores for correspondences
    """
    print("\n" + "="*60)
    print("Testing Threshold Sensitivity")
    print("="*60)
    
    thresholds = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    for th in thresholds:
        _, inl = estimator(pts0, pts1, th, ranking=ranking)
        count = inl.sum().item()
        ratio = 100 * count / len(pts0)
        print(f"Threshold {th:.1f}: {count:3d} inliers ({ratio:5.1f}%)")


def test_with_noise(estimator, pts0_clean, pts1_clean, ranking, noise_levels=[0.0, 0.5, 1.0, 2.0]):
    """
    Test PROSAC robustness to noise in correspondences.
    
    Args:
        estimator (PROSACRelativePoseEstimator): PROSAC estimator instance
        pts0_clean (torch.Tensor): Clean correspondences in image 1
        pts1_clean (torch.Tensor): Clean correspondences in image 2
        ranking (torch.Tensor): Ranking scores
        noise_levels (list): Standard deviations of Gaussian noise to test
    """
    print("\n" + "="*60)
    print("Testing Robustness to Noise")
    print("="*60)
    
    for noise in noise_levels:
        pts0_noisy = pts0_clean + noise * torch.randn_like(pts0_clean)
        pts1_noisy = pts1_clean + noise * torch.randn_like(pts1_clean)
        
        _, inl = estimator(pts0_noisy, pts1_noisy, 0.5, ranking=ranking)
        count = inl.sum().item()
        ratio = 100 * count / len(pts0_noisy)
        print(f"Noise σ={noise:.1f}: {count:3d} inliers ({ratio:5.1f}%)")


if __name__ == "__main__":
    print("="*60)
    print("PROSAC Stability and Robustness Tests")
    print("="*60)
    
    # Setup test data
    torch.manual_seed(0)
    
    # Create synthetic correspondences
    N = 200
    pts0 = torch.rand(N, 2) * 100  # Random points in 100x100 space
    pts1 = pts0 + 0.5 * torch.randn(N, 2)  # Add small noise
    
    # Create ranking based on noise (lower noise = higher rank)
    noise = torch.norm(pts1 - pts0, dim=1)
    ranking = -noise  # Negative because argsort in PROSAC uses descending=True
    
    # Create estimator
    estimator = PROSACRelativePoseEstimator(iterations=5000, irls_iters=10, seed=32000)
    
    # Test 1: Repeatability
    print("\n" + "="*60)
    print("Test 1: Repeatability (Deterministic with fixed seed)")
    print("="*60)
    test_repeatability(estimator, pts0, pts1, ranking, th=0.5, runs=5)
    
    # Test 2: Ranking effect
    test_ranking_effect(estimator, pts0, pts1, th=0.5)
    
    # Test 3: Threshold sensitivity
    test_threshold_sensitivity(estimator, pts0, pts1, ranking)
    
    # Test 4: Robustness to noise
    test_with_noise(estimator, pts0, pts1, ranking)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)