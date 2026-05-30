# src/calibration.py
"""
Stage 2: Calibration pipeline.

Maps anomaly score s(x) → predicted pose error ê(x) using a small labeled
HIL subset (~25% of each domain). Calibrators are fit separately for lightbox
and sunlamp because the two conditions exhibit different score-to-error curves.

Two calibrators per (detector, domain) pair:
  - IsotonicRegression:  s(x) → ehat(x)      continuous error prediction
  - LogisticRegression:  s(x) → P(fail)   binary fail/pass, threshold on SPEED score

Saves:
    results/calibration_results.csv  -- per-domain evaluation metrics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, f1_score, roc_auc_score,
    precision_score, recall_score,
)
from scipy.stats import spearmanr

PROJECT_ROOT   = Path(__file__).parent.parent
RESULTS_DIR    = PROJECT_ROOT / 'results'

HIL_DOMAINS    = ['lightbox', 'sunlamp']
DETECTORS      = ['mahal', 'gmm']
FAIL_THRESHOLD = 0.05   # SPEED score above this → failure
CALIB_FRAC     = 0.25   # fraction of HIL images reserved for calibration (~250 per domain)


def signed_log(scores: np.ndarray) -> np.ndarray:
    """
    Deal with heavy right tail of the anomaly scores via signed log transform.
    """
    return np.sign(scores) * np.log1p(np.abs(scores))


def load_scores_and_errors(detector: str, domain: str):
    """
    Load anomaly scores and SPEED errors for a HIL domain, aligned by row.
    Both arrays come from the test CSV in the same order; truncate to min length
    in case a small number of images failed during SPNv2 inference.
    """
    scores = np.load(RESULTS_DIR / f'anomaly_scores_{detector}_{domain}.npy')
    errors_df = pd.read_csv(RESULTS_DIR / f'per_image_errors_{domain}.csv')
    n = min(len(scores), len(errors_df))
    return signed_log(scores[:n]), errors_df['speed_score'].values[:n]


class DomainCalibrator:
    """
    Isotonic + logistic calibrators for one (detector, domain) pair.
    Fit on a small labeled calibration split; applied to the held-out test split.
    """

    def __init__(self, fail_threshold: float = FAIL_THRESHOLD):
        self.fail_threshold = fail_threshold
        self.iso_ = IsotonicRegression(increasing=True, out_of_bounds='clip')
        self.lr_  = LogisticRegression(class_weight='balanced', max_iter=1000,
                                       random_state=42)

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> 'DomainCalibrator':
        labels = (errors > self.fail_threshold).astype(int)
        self.iso_.fit(scores, errors)
        self.lr_.fit(scores.reshape(-1, 1), labels)
        return self

    def predict_error(self, scores: np.ndarray) -> np.ndarray:
        return self.iso_.predict(scores)

    def predict_fail_prob(self, scores: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(scores.reshape(-1, 1))[:, 1]

    def predict_fail(self, scores: np.ndarray) -> np.ndarray:
        return self.lr_.predict(scores.reshape(-1, 1))


def evaluate(calibrator: DomainCalibrator,
             scores: np.ndarray,
             errors: np.ndarray) -> dict:
    labels    = (errors > calibrator.fail_threshold).astype(int)
    e_hat     = calibrator.predict_error(scores)
    fail_prob = calibrator.predict_fail_prob(scores)
    fail_pred = calibrator.predict_fail(scores)
    rho, _    = spearmanr(scores, errors)

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


def main():
    rows = []

    for det in DETECTORS:
        for domain in HIL_DOMAINS:
            scores, errors = load_scores_and_errors(det, domain)
            labels = (errors > FAIL_THRESHOLD).astype(int)

            s_calib, s_test, e_calib, e_test = train_test_split(
                scores, errors,
                test_size=1 - CALIB_FRAC,
                random_state=42,
                stratify=labels,
            )

            print(f"\n── {det} / {domain} ──")
            print(f"  Calibration set: {len(s_calib)} images")
            print(f"  Test set:        {len(s_test)} images")
            print(f"  Fail rate (calib): {(e_calib > FAIL_THRESHOLD).mean():.1%}")

            cal = DomainCalibrator().fit(s_calib, e_calib)
            m   = evaluate(cal, s_test, e_test)

            print(f"  MAE:       {m['mae']:.4f}  (isotonic, continuous error)")
            print(f"  Spearman:  {m['spearman']:.3f}  (rank correlation: score vs error)")
            print(f"  AUC-ROC:   {m['auc']:.3f}  (logistic binary classifier)")
            print(f"  F1:        {m['f1']:.3f}")
            print(f"  Precision: {m['precision']:.3f}")
            print(f"  Recall:    {m['recall']:.3f}")
            print(f"  Fail rate (test): {m['fail_rate']:.1%}  ({m['n_fail']}/{m['n_total']})")

            rows.append({'detector': det, 'domain': domain, **m})

    df  = pd.DataFrame(rows)
    out = RESULTS_DIR / 'calibration_results.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved calibration metrics → {out.name}")
    print()
    print(df[['detector', 'domain', 'mae', 'spearman', 'auc', 'f1',
              'precision', 'recall', 'fail_rate']].to_string(index=False))


if __name__ == '__main__':
    main()
