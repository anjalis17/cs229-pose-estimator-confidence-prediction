"""
Dumb baselines for failure prediction, scored per error axis.

  random        flag failures at the synthetic training failure rate (HIL labels
                never seen). One constant risk score per image, so AUC = 0.5.
  disagreement  use the matching multi-head disagreement feature directly as the
                score (disagree_t_m for translation, disagree_R_deg for rotation).

AUC is threshold-free on the continuous disagreement score. The binary call does
NOT threshold at the physical error limit (internal head disagreement is not the
external pose error); instead we freeze an F1-optimal cutoff on a held-out
synthetic val split and apply it unchanged to HIL, matching the train/val
protocol of our method (model_outputs.synth_val_threshold). disagree_* is NaN on
PnP-rejected rows, so the HIL baselines are scored on reject==0 rows only.

Our actual method lives in models.py, not here.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, precision_score, recall_score,
)

PROJECT_ROOT = Path(__file__).parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'

DOMAINS = ['lightbox', 'sunlamp']

# axis -> error column, absolute fail threshold, and the disagreement score to use
AXES = {
    'translation': {'err_col': 'E_T', 'threshold': 0.10, 'feature': 'disagree_t_m'},
    'rotation': {'err_col': 'E_R', 'threshold': 3.0, 'feature': 'disagree_R_deg'},
}

# synthetic val split for freezing the disagreement cutoff; same fraction and seed
# as model_outputs.synth_val_threshold so baseline and method pick on identical data
VAL_FRAC = 0.25
RANDOM_STATE = 42


def load_domain(domain):
    # returns (cols, errors_df) aligned to a common length; cols maps each
    # model-feature name -> its (N,) column
    feats = np.load(RESULTS_DIR / f'model_features_{domain}.npy')
    names = [str(n) for n in np.load(RESULTS_DIR / 'model_feature_names.npy',
                                     allow_pickle=True)]
    df = pd.read_csv(RESULTS_DIR / f'per_image_errors_{domain}.csv')
    n = min(len(feats), len(df))
    feats, df = feats[:n], df.iloc[:n].reset_index(drop=True)
    cols = {nm: feats[:, j] for j, nm in enumerate(names)}
    return cols, df


def freeze_disagree_threshold(axis, cols_synth, df_synth):
    """F1-optimal disagreement cutoff on a held-out synthetic val split. Returns
    (threshold, synth_val_f1); HIL is never touched."""
    cfg = AXES[axis]
    ys = (df_synth[cfg['err_col']].values > cfg['threshold']).astype(int)
    score = cols_synth[cfg['feature']]
    keep = cols_synth['reject'].astype(int) == 0  # no-op on synthetic, kept for parity
    score, ys = score[keep], ys[keep]

    _, s_val, _, y_val = train_test_split(
        score, ys, test_size=VAL_FRAC, stratify=ys, random_state=RANDOM_STATE)

    # prec/rec have len(thr)+1; drop the trailing point to align f1 to thr
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    if not len(thr):
        return np.inf, np.nan
    best = int(np.nanargmax(f1))
    return float(thr[best]), float(f1[best])


def evaluate_axis(domain, axis, cols, df, t_disagree, synth_rate):
    cfg = AXES[axis]
    y = (df[cfg['err_col']].values > cfg['threshold']).astype(int)
    keep = cols['reject'].astype(int) == 0  # drop degenerate (imputed) rows
    n_drop = int((~keep).sum())

    yk = y[keep]
    auc_defined = len(np.unique(yk)) > 1  # AUC needs both classes present

    common = {'domain': domain, 'axis': axis,
              'n': int(keep.sum()), 'n_dropped_reject': n_drop,
              'fail_rate': float(yk.mean()) if len(yk) else np.nan}

    # random: in expectation flagged rate = recall = synth_rate, precision = HIL base rate
    random_row = {**common, 'baseline': 'random',
                  'auc': 0.5 if auc_defined else np.nan,
                  'precision': float(yk.mean()) if len(yk) else np.nan,
                  'recall': synth_rate,
                  'flagged_rate': synth_rate}

    # disagreement: feature value is the score; binary call uses the frozen val cutoff
    score = cols[cfg['feature']][keep]
    pred = (score >= t_disagree).astype(int)
    disag_row = {**common, 'baseline': 'disagreement',
                 'auc': roc_auc_score(yk, score) if auc_defined else np.nan,
                 'threshold': t_disagree,
                 'precision': precision_score(yk, pred, zero_division=0),
                 'recall': recall_score(yk, pred, zero_division=0),
                 'flagged_rate': float(pred.mean()) if len(pred) else np.nan}

    return [random_row, disag_row]


def main():
    # per-axis synthetic stats (computed once): random flag rate + frozen cutoff
    cols_synth, df_synth = load_domain('synthetic')
    thresholds, synth_rates = {}, {}
    print("Synthetic stats (no HIL labels touched):")
    for axis in AXES:
        cfg = AXES[axis]
        synth_rates[axis] = float((df_synth[cfg['err_col']].values > cfg['threshold']).mean())
        t, val_f1 = freeze_disagree_threshold(axis, cols_synth, df_synth)
        thresholds[axis] = t
        print(f"  {axis:11s} random flag rate = {synth_rates[axis]:.3f}   |   "
              f"{cfg['feature']} >= {t:.4f}  (synthetic-val F1 = {val_f1:.3f})")

    all_rows = []
    for domain in DOMAINS:
        cols, df = load_domain(domain)
        for axis in AXES:
            rows = evaluate_axis(domain, axis, cols, df,
                                 thresholds[axis], synth_rates[axis])
            all_rows += rows
            n_drop, n_keep = rows[0]['n_dropped_reject'], rows[0]['n']
            print(f"\n{domain} / {axis}: dropped {n_drop} reject==1 row(s) "
                  f"(disagree_* degenerate) -> {n_keep} rows, "
                  f"fail_rate={rows[0]['fail_rate']:.3f}")
            for r in rows:
                auc = f"{r['auc']:.3f}" if not np.isnan(r['auc']) else "nan (one class)"
                line = (f"    {r['baseline']:13s} AUC = {auc}"
                        f"  |  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                        f"flagged={r['flagged_rate']:.3f}")
                if r['baseline'] == 'disagreement':
                    line += f"  (t={r['threshold']:.4f})"
                print(line)

    df_out = pd.DataFrame(all_rows)[
        ['domain', 'axis', 'baseline', 'auc', 'precision', 'recall',
         'flagged_rate', 'threshold', 'fail_rate', 'n', 'n_dropped_reject']
    ]
    out = RESULTS_DIR / 'baseline_results.csv'
    df_out.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY  (vs absolute-threshold failure label)")
    print("  AUC: continuous disagreement score (threshold-free)")
    print("  P / R / flagged: frozen synthetic-val cutoff applied to HIL")
    print(f"{'='*60}")
    print(df_out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"\nSaved -> results/baseline_results.csv")


if __name__ == '__main__':
    main()
