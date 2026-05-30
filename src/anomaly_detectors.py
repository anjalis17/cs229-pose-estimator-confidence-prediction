# src/anomaly_detectors.py
"""
Stage 1: Unsupervised anomaly detectors fit on synthetic image features.

Each detector scores a test image by how far it is from the synthetic
training distribution. Higher score → more anomalous → expected higher
pose error after calibration in Stage 2.

Saves:
    results/anomaly_scores_mahal_{domain}.npy  -- (N,) Mahalanobis scores
    results/anomaly_scores_gmm_{domain}.npy    -- (N,) GMM NLL scores
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / 'results'

DOMAINS = ['synthetic', 'lightbox', 'sunlamp']


def load_features(domain: str) -> np.ndarray:
    # return np.load(RESULTS_DIR / f'img_stats_{domain}.npy')
    return np.load(RESULTS_DIR / f'model_features_{domain}.npy')


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
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                  random_state=42, max_iter=200)
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


# Train / test loop
# Train on synthetic features, then score all domains and save results for calibration pipeline
def main():
    X_synth = load_features('synthetic')
    print(f"Synthetic: {X_synth.shape[0]} images, {X_synth.shape[1]} features\n")

    detectors = {
        'mahal': MahalanobisDetector(),
        'gmm':   GMMDetector(),
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
