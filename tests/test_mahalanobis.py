"""
Tests for MahalanobisDetector in src/anomaly_detectors.py.

Covers:
  - Mathematical correctness (vs scipy ground truth)
  - Output shape and dtype
  - Boundary / degenerate inputs
  - Behavioral invariants (non-negativity, ordering, ridge)
  - Integration: real saved features produce expected domain ordering
"""

import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

from src.anomaly_detectors import MahalanobisDetector

RNG = np.random.default_rng(42)
RESULTS_DIR = Path(__file__).parent.parent / 'results'


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_data():
    """200 samples, 5 features drawn from a known Gaussian."""
    mu    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cov   = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    X     = RNG.multivariate_normal(mu, cov, size=200)
    return X, mu, cov


@pytest.fixture
def fitted_detector(simple_data):
    X, _, _ = simple_data
    return MahalanobisDetector().fit(X)


# ── Shape / dtype ─────────────────────────────────────────────────────────────

def test_score_shape(fitted_detector, simple_data):
    X, _, _ = simple_data
    scores = fitted_detector.score(X)
    assert scores.shape == (len(X),)


def test_score_single_sample(fitted_detector, simple_data):
    X, _, _ = simple_data
    scores = fitted_detector.score(X[:1])
    assert scores.shape == (1,)


def test_fit_returns_self():
    X = RNG.standard_normal((50, 4))
    det = MahalanobisDetector()
    result = det.fit(X)
    assert result is det


# ── Mathematical correctness ──────────────────────────────────────────────────

def test_score_at_mean_is_zero(fitted_detector):
    """Score of the fitted mean should be 0 (or numerically indistinguishable)."""
    mu = fitted_detector.mu_
    score = fitted_detector.score(mu.reshape(1, -1))
    assert score[0] == pytest.approx(0.0, abs=1e-8)


def test_scores_nonnegative(fitted_detector, simple_data):
    X, _, _ = simple_data
    scores = fitted_detector.score(X)
    assert np.all(scores >= -1e-10), f"Negative score found: {scores.min()}"


def test_matches_scipy(fitted_detector, simple_data):
    """
    MahalanobisDetector.score should match scipy_mahalanobis ** 2
    for each sample (scipy returns the distance, we return the squared distance).
    """
    X, _, _ = simple_data
    det = fitted_detector
    VI = det.cov_inv_

    our_scores = det.score(X[:10])
    scipy_scores = np.array([
        scipy_mahalanobis(x, det.mu_, VI) ** 2
        for x in X[:10]
    ])
    np.testing.assert_allclose(our_scores, scipy_scores, rtol=1e-5,
                               err_msg="Scores diverge from scipy reference")


def test_identity_covariance_equals_squared_euclidean():
    """
    With cov_inv = I and mu = 0, score should equal squared Euclidean distance.
    Directly inject known parameters to avoid relying on empirical convergence.
    """
    n_features = 6
    det = MahalanobisDetector()
    det.mu_      = np.zeros(n_features)
    det.cov_inv_ = np.eye(n_features)

    X_test      = RNG.standard_normal((20, n_features))
    our_scores  = det.score(X_test)
    eucl_scores = np.sum(X_test ** 2, axis=1)  # squared Euclidean from origin

    np.testing.assert_allclose(our_scores, eucl_scores, rtol=1e-10,
                               err_msg="With I covariance, Mahal ≠ squared Euclidean")


def test_known_2d_value():
    """
    Manually verify a single 2-D case with exact arithmetic.
    X = [[0, 0], [2, 0]] → mu = [1, 0], cov ≈ [[2, 0],[0, 0+ridge]]
    Point [3, 0]: diff = [2, 0], Mahal^2 = 4 / (2 + ridge) ≈ 2.
    """
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    det = MahalanobisDetector().fit(X)

    # empirical cov of two points: var([0,2]) = 2 (ddof=1), covar = 0
    # cov[0,0] = 2 + 1e-6, cov[1,1] = 0 + 1e-6
    cov_inv_expected = np.diag([1 / (2 + 1e-6), 1 / 1e-6])
    np.testing.assert_allclose(det.cov_inv_, cov_inv_expected, rtol=1e-5)

    point = np.array([[3.0, 0.0]])  # diff from mu=[1,0] is [2, 0]
    expected = 4.0 / (2 + 1e-6)
    score = det.score(point)
    assert score[0] == pytest.approx(expected, rel=1e-5)


# ── Behavioral invariants ─────────────────────────────────────────────────────

def test_symmetry_around_mean(fitted_detector):
    """Two points symmetric about the mean get the same score."""
    mu = fitted_detector.mu_
    delta = RNG.standard_normal(mu.shape)
    p1 = (mu + delta).reshape(1, -1)
    p2 = (mu - delta).reshape(1, -1)
    s1 = fitted_detector.score(p1)
    s2 = fitted_detector.score(p2)
    assert s1[0] == pytest.approx(s2[0], rel=1e-6)


def test_farther_point_scores_higher(fitted_detector):
    """A point 3× farther from the mean in the same direction scores higher."""
    mu    = fitted_detector.mu_
    delta = RNG.standard_normal(mu.shape)
    delta /= np.linalg.norm(delta)

    close = (mu + 1.0 * delta).reshape(1, -1)
    far   = (mu + 3.0 * delta).reshape(1, -1)
    assert fitted_detector.score(far)[0] > fitted_detector.score(close)[0]


def test_ridge_prevents_singular_crash():
    """
    A constant (zero-variance) feature column makes cov rank-deficient.
    The ridge should prevent a crash and still produce finite scores.
    """
    X = RNG.standard_normal((50, 4))
    X[:, 2] = 5.0           # constant column → singular covariance

    det = MahalanobisDetector().fit(X)
    scores = det.score(X)
    assert np.all(np.isfinite(scores)), "Scores should be finite even with singular cov"


def test_ood_scores_higher_than_in_distribution():
    """
    Points drawn far from the training distribution score higher on average
    than in-distribution points.
    """
    X_train = RNG.multivariate_normal(np.zeros(8), np.eye(8), size=300)
    X_in    = RNG.multivariate_normal(np.zeros(8), np.eye(8), size=100)
    X_ood   = RNG.multivariate_normal(10 * np.ones(8), np.eye(8), size=100)

    det = MahalanobisDetector().fit(X_train)
    assert det.score(X_ood).mean() > det.score(X_in).mean()


# ── Integration: real saved features ─────────────────────────────────────────

@pytest.mark.skipif(
    not (RESULTS_DIR / 'img_stats_synthetic.npy').exists(),
    reason="Pre-computed feature files not present"
)
def test_real_data_domain_ordering():
    """
    Synthetic median score < lightbox median < sunlamp median.
    This validates that the detector captures the expected domain shift.
    """
    X_synth   = np.load(RESULTS_DIR / 'img_stats_synthetic.npy')
    X_lb      = np.load(RESULTS_DIR / 'img_stats_lightbox.npy')
    X_sun     = np.load(RESULTS_DIR / 'img_stats_sunlamp.npy')

    det = MahalanobisDetector().fit(X_synth)

    median_synth = np.median(det.score(X_synth))
    median_lb    = np.median(det.score(X_lb))
    median_sun   = np.median(det.score(X_sun))

    assert median_synth < median_lb, (
        f"Synthetic median ({median_synth:.1f}) should be < lightbox ({median_lb:.1f})"
    )
    assert median_lb < median_sun, (
        f"Lightbox median ({median_lb:.1f}) should be < sunlamp ({median_sun:.1f})"
    )


@pytest.mark.skipif(
    not (RESULTS_DIR / 'anomaly_scores_mahal_synthetic.npy').exists(),
    reason="Pre-computed score files not present (run anomaly_detectors.main() first)"
)
def test_saved_scores_match_fresh_computation():
    """Scores saved to disk by main() must match a fresh fit+score."""
    X_synth = np.load(RESULTS_DIR / 'img_stats_synthetic.npy')
    det     = MahalanobisDetector().fit(X_synth)

    for domain in ['synthetic', 'lightbox', 'sunlamp']:
        X      = np.load(RESULTS_DIR / f'img_stats_{domain}.npy')
        saved  = np.load(RESULTS_DIR / f'anomaly_scores_mahal_{domain}.npy')
        fresh  = det.score(X)
        np.testing.assert_allclose(
            fresh, saved, rtol=1e-5,
            err_msg=f"Saved scores for {domain} don't match fresh computation"
        )
