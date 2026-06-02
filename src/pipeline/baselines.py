# src/pipeline/baselines.py
"""
Dumb baselines for failure prediction, evaluated per error axis.

Two baselines, each scored by AUC against an ABSOLUTE-threshold failure label,
independently for translation and rotation:

  - random        — predict failure at the domain's empirical failure rate.
                    A constant score, so AUC = 0.5 by construction (the floor).
  - disagreement  — use the matching multi-head disagreement feature directly as
                    the failure score (no classifier, no cutoff; the feature value
                    IS y_prob):
                        translation → disagree_t_m
                        rotation    → disagree_R_deg

Absolute failure thresholds (a fail is a fail, domain-independent):
    translation error  E_T > 0.10 m
    rotation    error  E_R > 3.0°

disagree_* is NaN on PnP-rejected rows (median-imputed upstream), which is
degenerate for this baseline. Both baselines are therefore scored on reject == 0
rows only, and the number of dropped rows is reported.

Our actual method (supervised + importance-weighted classifiers) lives in
src/pipeline/models.py — not here.

Saves:
    results/baseline_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).parents[2]
RESULTS_DIR  = PROJECT_ROOT / 'results'

DOMAINS = ['lightbox', 'sunlamp']

# axis → error column, absolute fail threshold, and the disagreement score to use
AXES = {
    'translation': {'err_col': 'E_T', 'threshold': 0.10, 'feature': 'disagree_t_m'},
    'rotation':    {'err_col': 'E_R', 'threshold': 3.0,  'feature': 'disagree_R_deg'},
}


# ── Data ──────────────────────────────────────────────────────────────────────
def load_domain(domain: str):
    """Return (cols, errors_df) aligned to a common length.
    cols maps each model-feature name → its (N,) column."""
    feats = np.load(RESULTS_DIR / f'model_features_{domain}.npy')
    names = [str(n) for n in np.load(RESULTS_DIR / 'model_feature_names.npy',
                                     allow_pickle=True)]
    df    = pd.read_csv(RESULTS_DIR / f'per_image_errors_{domain}.csv')
    n     = min(len(feats), len(df))
    feats, df = feats[:n], df.iloc[:n].reset_index(drop=True)
    cols  = {nm: feats[:, j] for j, nm in enumerate(names)}
    return cols, df


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_axis(domain: str, axis: str, cols: dict, df: pd.DataFrame):
    """Random + disagreement baselines for one (domain, axis), scored by AUC on
    the reject == 0 subset."""
    cfg    = AXES[axis]
    y      = (df[cfg['err_col']].values > cfg['threshold']).astype(int)
    keep   = cols['reject'].astype(int) == 0           # drop degenerate (imputed) rows
    n_drop = int((~keep).sum())

    yk          = y[keep]
    auc_defined = len(np.unique(yk)) > 1               # AUC needs both classes present

    common = {'domain': domain, 'axis': axis,
              'n': int(keep.sum()), 'n_dropped_reject': n_drop,
              'fail_rate': float(yk.mean()) if len(yk) else np.nan}

    # random: constant score = empirical failure rate → AUC = 0.5
    random_row = {**common, 'baseline': 'random',
                  'auc': 0.5 if auc_defined else np.nan}

    # disagreement: the feature value is the score (higher = more likely to fail)
    score = cols[cfg['feature']][keep]
    disag_row = {**common, 'baseline': 'disagreement',
                 'auc': roc_auc_score(yk, score) if auc_defined else np.nan}

    return [random_row, disag_row]


def main():
    all_rows = []
    for domain in DOMAINS:
        cols, df = load_domain(domain)
        for axis in AXES:
            rows = evaluate_axis(domain, axis, cols, df)
            all_rows += rows
            n_drop, n_keep = rows[0]['n_dropped_reject'], rows[0]['n']
            print(f"{domain} / {axis}: dropped {n_drop} reject==1 row(s) "
                  f"(disagree_* degenerate) → {n_keep} rows, "
                  f"fail_rate={rows[0]['fail_rate']:.3f}")
            for r in rows:
                auc = f"{r['auc']:.3f}" if not np.isnan(r['auc']) else "nan (one class)"
                print(f"    {r['baseline']:13s} AUC = {auc}")

    df_out = pd.DataFrame(all_rows)[
        ['domain', 'axis', 'baseline', 'auc', 'fail_rate', 'n', 'n_dropped_reject']
    ]
    out = RESULTS_DIR / 'baseline_results.csv'
    df_out.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY  (AUC vs absolute-threshold failure label)")
    print(f"{'='*60}")
    print(df_out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"\nSaved → results/baseline_results.csv")


if __name__ == '__main__':
    main()
