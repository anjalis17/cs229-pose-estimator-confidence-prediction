# src/anomaly_detectors.py
"""
Stage 1: Unsupervised anomaly detectors fit on synthetic image features.

Each detector scores a test image by how far it is from the synthetic
training distribution. Higher score → more anomalous → expected higher
pose error after calibration in Stage 2.

Saves:
    results/anomaly_scores_mahal_{domain}.npy  -- (N,) Mahalanobis scores
    results/anomaly_scores_gmm_{domain}.npy    -- (N,) GMM NLL scores
    results/anomaly_scores_ocsvm_{domain}.npy  -- (N,) One-Class SVM margin scores
    results/anomaly_scores_iforest_{domain}.npy -- (N,) Isolation Forest path-length scores
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / 'results'

DOMAINS = ['synthetic', 'lightbox', 'sunlamp']

# Features dropped before fitting any detector.
# 'reject' is constant (=0) on all synthetic images → zero variance. It carries no
# in-distribution signal and actively corrupts the Mahalanobis score: with the 1e-6
# ridge, a HIL image with reject=1 contributes ~(1-0)^2 / 1e-6 ≈ 1e6 to the squared
# distance, swamping the real signal for the handful of rejected HIL images.
DROP_FEATURES = ['reject']

# 13 saved feature names; we keep all but DROP_FEATURES (→ 12 features).
ALL_FEATURE_NAMES = np.load(RESULTS_DIR / 'model_feature_names.npy', allow_pickle=True).tolist()
KEEP_IDX   = [i for i, n in enumerate(ALL_FEATURE_NAMES) if n not in DROP_FEATURES]
KEEP_NAMES = [ALL_FEATURE_NAMES[i] for i in KEEP_IDX]


def load_features(domain: str) -> np.ndarray:
    """Load the model features for a domain, dropping DROP_FEATURES (→ 12 columns)."""
    X = np.load(RESULTS_DIR / f'model_features_{domain}.npy')
    return X[:, KEEP_IDX]


# ================================================ Mahalanobis Detector ================================================

# Mahalanobis distance is the distance from a point to the mean of the Gaussian, measured in units of standard deviations
# In a multivariate sense, it also accounts for correlations between features and the individual variances of each feature
class MahalanobisDetector:
    """
    Fits a single Gaussian (μ, Σ) to synthetic features.
    Anomaly score: (x - μ)ᵀ Σ⁻¹ (x - μ).
    """

    def fit(self, X: np.ndarray) -> 'MahalanobisDetector':
        self.mu_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)  # each column is a feature, each row is an image
        # small ridge to guard against near-singularity
        cov += 1e-6 * np.eye(cov.shape[0])
        self.cov_inv_ = np.linalg.inv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) Mahalanobis distances."""
        diff = X - self.mu_
        # This computes the scalar diff[n] @ cov_inv @ diff[n] independently for each sample n
        # This is the squared Mahalanobis distance -> will be used for anomaly scoring (higher = more anomalous)
        return np.einsum('ni,ij,nj->n', diff, self.cov_inv_, diff)


# ================================================ GMM Detector ================================================

class GMMDetector:
    """
    Fits a Gaussian Mixture Model to synthetic features via EM.
    Anomaly score: -log p(x) under the mixture (higher score = lower probability = more anomalous).

    K is selected automatically by minimizing BIC over a candidate set,
    which balances fit quality against model complexity.

    Features are standardized before fitting so that no single feature
    dominates the covariance structure due to scale differences.
    """

    K_CANDIDATES = [1, 2, 4, 8, 16]

    def fit(self, X: np.ndarray) -> 'GMMDetector':
        self.scaler_ = StandardScaler().fit(X)
        X_s = self.scaler_.transform(X)

        # select K (num Gaussians) by BIC — lower is better
        # BIC is Bayesian Information Criteria (BIC = -2 * log-likelihood + k * log(n));
        # - k is the number of model params and n is the number of samples.
        # - penalizes model complexity (more params) to avoid overfitting, while rewarding better fit (higher log-likelihood).
        best_bic, best_gmm = np.inf, None
        print("  GMM BIC selection:")
        for k in self.K_CANDIDATES:
            # reg_covar adds a small ridge to each component covariance so EM
            # stays numerically stable: without it a component can collapse onto
            # a near-degenerate direction (e.g. the discrete hm_nconf) at high K
            # and make the covariance non-positive-definite.
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                  reg_covar=1e-4, random_state=42, max_iter=200)
            gmm.fit(X_s)
            bic = gmm.bic(X_s)
            print(f"    K={k:>2}  BIC={bic:.1f}")
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm

        self.gmm_    = best_gmm
        self.best_k_ = best_gmm.n_components
        print(f"  → Selected K={self.best_k_}  (BIC={best_bic:.1f})")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) negative log-likelihoods under the fitted mixture."""
        X_s = self.scaler_.transform(X)
        # Computes the density p(x) for each sample x under the fitted GMM, 
        # then returns -log p(x) as the anomaly score (higher = more anomalous)
        return -self.gmm_.score_samples(X_s)


# ================================================ One-Class SVM Detector ================================================

class OCSVMDetector:
    """
    Fits a One-Class SVM with an RBF kernel to synthetic features.

    Unlike Mahalanobis (single Gaussian) and GMM (mixture of Gaussians), the
    OC-SVM makes no parametric distributional assumption: it learns a tight
    boundary around the synthetic support in RBF feature space, so it serves as
    a complementary *nonparametric* baseline.

    Hyperparameters (no unsupervised criterion like GMM's BIC, so set by default):
      - nu:    upper bound on the fraction of synthetic points allowed outside
               the boundary / lower bound on the fraction of support vectors.
               Small, since synthetic is our "clean" reference distribution.
      - gamma: RBF kernel width. 'scale' = 1 / (n_features * X.var()).

    Features are standardized before fitting because the RBF kernel is purely
    distance-based — without it a large-scale feature would dominate the kernel.

    Anomaly score: sklearn's decision_function is positive for inliers and
    negative for outliers, so we return its negation to match the project-wide
    "higher = more anomalous" convention.
    """

    def __init__(self, nu: float = 0.05, gamma='scale'):
        self.nu = nu
        self.gamma = gamma

    def fit(self, X: np.ndarray) -> 'OCSVMDetector':
        self.scaler_ = StandardScaler().fit(X)
        X_s = self.scaler_.transform(X)
        self.svm_ = OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)
        self.svm_.fit(X_s)
        print(f"  OC-SVM: nu={self.nu}, gamma={self.gamma}, "
              f"{self.svm_.support_vectors_.shape[0]} support vectors")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) anomaly scores: negated signed margin (higher = more anomalous)."""
        X_s = self.scaler_.transform(X)
        # decision_function > 0 inside the boundary (inlier), < 0 outside (outlier);
        # negate so larger values mean more anomalous, consistent with mahal/gmm.
        return -self.svm_.decision_function(X_s)


# ================================================ Isolation Forest Detector ================================================

class IsolationForestDetector:
    """
    Fits an Isolation Forest to synthetic features.

    Unlike Mahalanobis (single Gaussian) and GMM (mixture of Gaussians), it makes
    no distributional assumption: it builds an ensemble of random binary trees,
    each isolating points via random feature/threshold splits. Points that are
    isolated in few splits (short average path length) sit in sparse regions and
    are flagged anomalous; points buried in dense regions need many splits.

    Because splits are axis-aligned it captures feature *interactions* a single
    Gaussian misses, but is weaker along correlated/diagonal directions — making
    it a complementary signal to the distance-based Mahalanobis detector.

    Hyperparameters (no unsupervised criterion like GMM's BIC, so set by default):
      - n_estimators:  number of random trees; averaging stabilizes the score.
      - contamination: only affects sklearn's offset_/predict() boundary, not the
                       continuous score we return, so 'auto' is fine.

    Splits use only feature *ordering*, so standardization is not strictly needed,
    but we keep it for consistency with the other detectors.

    Anomaly score: sklearn's score_samples is higher for inliers (longer paths),
    so we return its negation to match the project-wide "higher = more anomalous".
    """

    def __init__(self, n_estimators: int = 200, contamination='auto'):
        self.n_estimators = n_estimators
        self.contamination = contamination

    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        self.scaler_ = StandardScaler().fit(X)
        X_s = self.scaler_.transform(X)
        self.iforest_ = IsolationForest(n_estimators=self.n_estimators,
                                        contamination=self.contamination,
                                        random_state=42)
        self.iforest_.fit(X_s)
        print(f"  Isolation Forest: n_estimators={self.n_estimators}, "
              f"contamination={self.contamination}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) anomaly scores: negated path-length score (higher = more anomalous)."""
        X_s = self.scaler_.transform(X)
        # score_samples is higher for inliers (longer isolation paths);
        # negate so larger values mean more anomalous, consistent with mahal/gmm.
        return -self.iforest_.score_samples(X_s)


# Train / test loop
# Train on synthetic features, then score all domains and save results for calibration pipeline
def main():
    X_synth = load_features('synthetic')
    print(f"Synthetic: {X_synth.shape[0]} images, {X_synth.shape[1]} features")
    print(f"Dropped {DROP_FEATURES} → keeping {len(KEEP_NAMES)}: {KEEP_NAMES}\n")

    # persist the kept feature names for the supervised pipeline / plots
    np.save(RESULTS_DIR / 'model_feature_names_kept.npy', np.array(KEEP_NAMES, dtype=object))

    detectors = {
        'mahal':   MahalanobisDetector(),
        'gmm':     GMMDetector(),
        'ocsvm':   OCSVMDetector(),
        'iforest': IsolationForestDetector(),
    }

    for name, detector in detectors.items():
        print(f"── Fitting {name} ──")
        detector.fit(X_synth)

        rows = []
        for domain in DOMAINS:
            X      = load_features(domain)
            scores = detector.score(X)

            out_path = RESULTS_DIR / f'anomaly_scores_{name}_{domain}.npy'
            np.save(out_path, scores)

            rows.append({
                'domain': domain,
                'n':      len(scores),
                'mean':   scores.mean(),
                'std':    scores.std(),
                'p25':    np.percentile(scores, 25),
                'median': np.median(scores),
                'p75':    np.percentile(scores, 75),
                'p95':    np.percentile(scores, 95),
            })
            print(f"  Saved {len(scores)} scores → {out_path.name}")

        print(f"\n{'='*65}")
        print(f"{name} score distribution (higher = more anomalous)")
        print(f"{'='*65}")
        df = pd.DataFrame(rows).set_index('domain')
        print(df[['n', 'mean', 'std', 'p25', 'median', 'p75', 'p95']].to_string())
        print()


if __name__ == '__main__':
    main()
