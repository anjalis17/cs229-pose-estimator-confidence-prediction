"""
Unsupervised anomaly detectors fit on synthetic image features.

Each detector scores a test image by how far it sits from the synthetic training
distribution (higher = more anomalous). Four detectors: Mahalanobis (single
Gaussian), GMM (mixture), one-class SVM and isolation forest (nonparametric).
Scores are saved per detector x domain for the downstream calibration pipeline.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'

DOMAINS = ['synthetic', 'lightbox', 'sunlamp']

# 'reject' is constant (=0) on synthetic so it carries no in-distribution signal,
# and with the Mahalanobis ridge a HIL reject=1 row contributes ~1e6 to the
# squared distance and swamps the real signal. Drop it.
DROP_FEATURES = ['reject']

ALL_FEATURE_NAMES = np.load(RESULTS_DIR / 'model_feature_names.npy', allow_pickle=True).tolist()
KEEP_IDX = [i for i, n in enumerate(ALL_FEATURE_NAMES) if n not in DROP_FEATURES]
KEEP_NAMES = [ALL_FEATURE_NAMES[i] for i in KEEP_IDX]


def load_features(domain):
    X = np.load(RESULTS_DIR / f'model_features_{domain}.npy')
    return X[:, KEEP_IDX]


class MahalanobisDetector:
    """Single Gaussian (mu, Sigma) on synthetic; score = (x-mu)^T Sigma^-1 (x-mu)."""

    def fit(self, X):
        self.mu_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov += 1e-6 * np.eye(cov.shape[0])  # ridge against near-singularity
        self.cov_inv_ = np.linalg.inv(cov)
        return self

    def score(self, X):
        diff = X - self.mu_
        # squared Mahalanobis distance per sample n: diff[n] @ cov_inv @ diff[n]
        return np.einsum('ni,ij,nj->n', diff, self.cov_inv_, diff)


class GMMDetector:
    """Gaussian mixture on synthetic via EM; score = -log p(x). K picked by BIC."""

    K_CANDIDATES = [1, 2, 4, 8, 16]

    def fit(self, X):
        self.scaler_ = StandardScaler().fit(X)
        X_s = self.scaler_.transform(X)

        # BIC = -2*loglik + k*log(n): penalizes complexity, lower is better
        best_bic, best_gmm = np.inf, None
        print("  GMM BIC selection:")
        for k in self.K_CANDIDATES:
            # reg_covar keeps EM stable: without it a component can collapse onto a
            # near-degenerate direction (e.g. discrete hm_nconf) at high K
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                  reg_covar=1e-4, random_state=42, max_iter=200)
            gmm.fit(X_s)
            bic = gmm.bic(X_s)
            print(f"    K={k:>2}  BIC={bic:.1f}")
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm

        self.gmm_ = best_gmm
        self.best_k_ = best_gmm.n_components
        print(f"  -> selected K={self.best_k_}  (BIC={best_bic:.1f})")
        return self

    def score(self, X):
        X_s = self.scaler_.transform(X)
        return -self.gmm_.score_samples(X_s)


class OCSVMDetector:
    """One-class SVM (RBF) on synthetic: a nonparametric tight boundary around the
    synthetic support. nu = fraction allowed outside; standardize since the RBF
    kernel is distance-based."""

    def __init__(self, nu=0.05, gamma='scale'):
        self.nu = nu
        self.gamma = gamma

    def fit(self, X):
        self.scaler_ = StandardScaler().fit(X)
        X_s = self.scaler_.transform(X)
        self.svm_ = OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)
        self.svm_.fit(X_s)
        print(f"  OC-SVM: nu={self.nu}, gamma={self.gamma}, "
              f"{self.svm_.support_vectors_.shape[0]} support vectors")
        return self

    def score(self, X):
        X_s = self.scaler_.transform(X)
        # decision_function > 0 inside the boundary; negate for "higher = more anomalous"
        return -self.svm_.decision_function(X_s)


class IsolationForestDetector:
    """Isolation forest on synthetic: random axis-aligned splits; points isolated
    in few splits (short path) are anomalous. Captures feature interactions a
    single Gaussian misses, complementary to Mahalanobis."""

    def __init__(self, n_estimators=200, contamination='auto'):
        self.n_estimators = n_estimators
        self.contamination = contamination

    def fit(self, X):
        self.scaler_ = StandardScaler().fit(X)
        X_s = self.scaler_.transform(X)
        self.iforest_ = IsolationForest(n_estimators=self.n_estimators,
                                        contamination=self.contamination,
                                        random_state=42)
        self.iforest_.fit(X_s)
        print(f"  Isolation Forest: n_estimators={self.n_estimators}, "
              f"contamination={self.contamination}")
        return self

    def score(self, X):
        X_s = self.scaler_.transform(X)
        # score_samples is higher for inliers (longer paths); negate to match convention
        return -self.iforest_.score_samples(X_s)


def main():
    X_synth = load_features('synthetic')
    print(f"Synthetic: {X_synth.shape[0]} images, {X_synth.shape[1]} features")
    print(f"Dropped {DROP_FEATURES} -> keeping {len(KEEP_NAMES)}: {KEEP_NAMES}\n")

    np.save(RESULTS_DIR / 'model_feature_names_kept.npy', np.array(KEEP_NAMES, dtype=object))

    detectors = {
        'mahal': MahalanobisDetector(),
        'gmm': GMMDetector(),
        'ocsvm': OCSVMDetector(),
        'iforest': IsolationForestDetector(),
    }

    for name, detector in detectors.items():
        print(f"-- Fitting {name} --")
        detector.fit(X_synth)

        rows = []
        for domain in DOMAINS:
            X = load_features(domain)
            scores = detector.score(X)

            out_path = RESULTS_DIR / f'anomaly_scores_{name}_{domain}.npy'
            np.save(out_path, scores)

            rows.append({
                'domain': domain,
                'n': len(scores),
                'mean': scores.mean(),
                'std': scores.std(),
                'p25': np.percentile(scores, 25),
                'median': np.median(scores),
                'p75': np.percentile(scores, 75),
                'p95': np.percentile(scores, 95),
            })
            print(f"  Saved {len(scores)} scores -> {out_path.name}")

        print(f"\n{'='*65}")
        print(f"{name} score distribution (higher = more anomalous)")
        print(f"{'='*65}")
        df = pd.DataFrame(rows).set_index('domain')
        print(df[['n', 'mean', 'std', 'p25', 'median', 'p75', 'p95']].to_string())
        print()


if __name__ == '__main__':
    main()
