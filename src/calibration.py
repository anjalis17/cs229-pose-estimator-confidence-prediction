# src/calibration.py
"""
Stage 2: Calibration pipeline.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, f1_score, roc_auc_score,
    precision_score, recall_score,
)
from scipy.stats import spearmanr

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

HIL_DOMAINS = ['lightbox', 'sunlamp']
REPRESENTATIONS = ['mahal', 'gmm', 'scores', 'features', 'features+scores']
CALIB_FRAC = 0.25   # fraction of HIL images reserved for calibration

# Per-component thresholds for rotation and translation errors [deg, m]
COMPONENTS = {'E_R': 5.0, 'E_T': 0.10}

# ── Data loading ──────────────────────────────────────────────────────────────
def signed_log(scores: np.ndarray) -> np.ndarray:
    """Tame the heavy right tail of the anomaly scores via signed log transform."""
    return np.sign(scores) * np.log1p(np.abs(scores))


def load_domain(domain: str):
    """
    Load every Stage-1 signal + per-component SPEED errors for one HIL domain,
    aligned by row.
    Returns {representation: (N, d) design matrix} and {component: (N,) errors}.

    All arrays come from the same test-CSV order; truncate to the min length in
    case a few images failed during SPNv2 inference.
    """
    feats  = np.load(RESULTS_DIR / f'model_features_{domain}.npy')
    mahal  = signed_log(np.load(RESULTS_DIR / f'anomaly_scores_mahal_{domain}.npy'))
    gmm    = signed_log(np.load(RESULTS_DIR / f'anomaly_scores_gmm_{domain}.npy'))
    err_df = pd.read_csv(RESULTS_DIR / f'per_image_errors_{domain}.csv')

    n = min(len(feats), len(mahal), len(gmm), len(err_df))
    feats, mahal, gmm, err_df = feats[:n], mahal[:n], gmm[:n], err_df.iloc[:n]
    scores = np.column_stack([mahal, gmm])

    X = {
        'mahal': mahal.reshape(-1, 1),
        'gmm': gmm.reshape(-1, 1),
        'scores': scores,
        'features': feats,
        'features+scores': np.column_stack([feats, scores]),
    }
    errors = {comp: err_df[comp].values for comp in COMPONENTS}
    return X, errors


# ── Calibrator ────────────────────────────────────────────────────────────────
class DomainCalibrator:
    """
    Standardize -> logistic (X -> P(fail)) + isotonic (P(fail) -> ehat) for one
    (component, representation, domain). Inputs can be any dimensionality, so the
    same calibrator works for a single anomaly score or the full feature vector.
    The isotonic output (ehat) is the per-measurement std-dev for the UKF;
    predict_fail_prob is the gating signal.
    """

    def __init__(self, fail_threshold: float):
        self.fail_threshold = fail_threshold
        self.scaler_ = StandardScaler()
        self.lr_  = LogisticRegression(class_weight='balanced', max_iter=1000,
                                       random_state=42)
        self.iso_ = IsotonicRegression(increasing=True, out_of_bounds='clip')

    def fit(self, X: np.ndarray, errors: np.ndarray) -> 'DomainCalibrator':
        labels = (errors > self.fail_threshold).astype(int)
        Xs = self.scaler_.fit_transform(X)
        self.lr_.fit(Xs, labels)
        probs = self.lr_.predict_proba(Xs)[:, 1]

        # isotonic maps the classifier's probability (a scalar) to continuous error,
        # so continuous prediction works regardless of input dimensionality.
        self.iso_.fit(probs, errors)

        # pick the probability threshold that maximizes F1 on the calibration split;
        # 0.5 is the wrong cut when the base failure rate isn't 50%
        grid = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(labels, (probs >= t).astype(int), zero_division=0) for t in grid]
        self.threshold_ = grid[int(np.argmax(f1s))]
        return self

    def predict_fail_prob(self, X: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(self.scaler_.transform(X))[:, 1]

    def predict_fail(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_fail_prob(X) >= self.threshold_).astype(int)

    def predict_error(self, X: np.ndarray) -> np.ndarray:
        return self.iso_.predict(self.predict_fail_prob(X))


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(cal: DomainCalibrator, X: np.ndarray, errors: np.ndarray) -> dict:
    labels    = (errors > cal.fail_threshold).astype(int)
    e_hat     = cal.predict_error(X)
    fail_prob = cal.predict_fail_prob(X)
    fail_pred = cal.predict_fail(X)
    rho, _    = spearmanr(fail_prob, errors)

    metrics = {
        'mae':       mean_absolute_error(errors, e_hat),
        'spearman':  rho,
        'n_fail':    int(labels.sum()),
        'n_total':   len(labels),
        'fail_rate': labels.mean(),
    }
    if len(np.unique(labels)) > 1:
        metrics['auc'] = roc_auc_score(labels, fail_prob)
        metrics['f1'] = f1_score(labels, fail_pred, zero_division=0)
        metrics['precision'] = precision_score(labels, fail_pred, zero_division=0)
        metrics['recall'] = recall_score(labels, fail_pred, zero_division=0)
    else:
        metrics.update({'auc': float('nan'), 'f1': float('nan'),
                        'precision': float('nan'), 'recall': float('nan')})
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rows = []

    for domain in HIL_DOMAINS:
        X, errors = load_domain(domain)

        for comp, threshold in COMPONENTS.items():
            err = errors[comp]
            labels = (err > threshold).astype(int)
            print(f"\n{'='*64}\n{domain} / {comp}: {len(err)} images, "
                  f"fail rate {labels.mean():.1%} (>{threshold} {('deg' if comp=='E_R' else 'm')})\n{'='*64}")

            # one split per (domain, component), shared across representations
            idx = np.arange(len(err))
            tr_idx, te_idx = train_test_split(
                idx, test_size=1 - CALIB_FRAC, random_state=42, stratify=labels,
            )

            for rep in REPRESENTATIONS:
                Xr = X[rep]
                cal = DomainCalibrator(fail_threshold=threshold).fit(Xr[tr_idx], err[tr_idx])
                m = evaluate(cal, Xr[te_idx], err[te_idx])

                print(f"\n── {rep}  ({Xr.shape[1]} feat) ──")
                print(f"  AUC:       {m['auc']:.3f}")
                print(f"  F1:        {m['f1']:.3f}   P {m['precision']:.3f}   R {m['recall']:.3f}")
                print(f"  MAE:       {m['mae']:.4f}   Spearman {m['spearman']:.3f}")

                rows.append({'domain': domain, 'component': comp, 'representation': rep,
                             'n_feat': Xr.shape[1], **m})

    df  = pd.DataFrame(rows)
    out = RESULTS_DIR / 'calibration_results.csv'
    df.to_csv(out, index=False)
    cols = ['domain', 'component', 'representation', 'n_feat', 'auc', 'f1',
            'precision', 'recall', 'mae', 'spearman']
    print(f"\n{'='*64}\nSUMMARY\n{'='*64}")
    print(df[cols].to_string(index=False))
    print(f"\nSaved → {out.name}")


if __name__ == '__main__':
    main()
