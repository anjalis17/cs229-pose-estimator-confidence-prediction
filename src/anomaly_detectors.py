# src/anomaly_detectors.py
"""
Stage 1: Unsupervised anomaly detectors fit on synthetic image features.

Each detector scores a test image by how far it is from the synthetic
training distribution. Higher score → more anomalous → expected higher
pose error after calibration in Stage 2.

Saves:
    results/anomaly_scores_mahal_{domain}.npy  -- (N,) Mahalanobis scores
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / 'results'

DOMAINS = ['synthetic', 'lightbox', 'sunlamp']


def load_features(domain: str) -> np.ndarray:
    return np.load(RESULTS_DIR / f'img_stats_{domain}.npy')


# ── Mahalanobis Detector ──────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_synth = load_features('synthetic')

    detector = MahalanobisDetector().fit(X_synth)
    print(f"Fit Mahalanobis detector on synthetic: {X_synth.shape[0]} images, "
          f"{X_synth.shape[1]} features")

    rows = []
    for domain in DOMAINS:
        X = load_features(domain)
        scores = detector.score(X)

        out_path = RESULTS_DIR / f'anomaly_scores_mahal_{domain}.npy'
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
        print(f"Saved {len(scores)} scores → {out_path.name}")

    print(f"\n{'='*65}")
    print("Mahalanobis score distribution (higher = more anomalous)")
    print(f"{'='*65}")
    df = pd.DataFrame(rows).set_index('domain')
    print(df[['n', 'mean', 'std', 'p25', 'median', 'p75', 'p95']].to_string())
    print()
    print("Expected: lightbox/sunlamp scores >> synthetic scores")


if __name__ == '__main__':
    main()
