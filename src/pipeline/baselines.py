# src/pipeline/baselines.py
"""
Dumb baselines for failure prediction, evaluated per error axis.

Two baselines, scored independently for translation and rotation against an
ABSOLUTE-threshold failure label:

  - random        — flag failures at the SYNTHETIC training failure rate (the
                    deployable prior; HIL labels are never seen). As a ranker it
                    assigns one constant risk score to every image, so AUC = 0.5
                    by construction (the floor). Its hard-decision metrics are the
                    expectation of independent Bernoulli(synth_rate) flagging:
                    flagged rate and recall both equal the synthetic failure rate,
                    precision equals the HIL base rate.
  - disagreement  — use the matching multi-head disagreement feature directly as
                    the failure score (no classifier; the feature value IS y_prob):
                        translation → disagree_t_m
                        rotation    → disagree_R_deg

Absolute failure thresholds (a fail is a fail, domain-independent):
    translation error  E_T > 0.10 m
    rotation    error  E_R > 3.0°

Two things are reported for the disagreement baseline:

  - AUC          — threshold-free, on the continuous disagreement score. Ranking
                   quality only; unaffected by any cutoff.
  - Precision / Recall / Flagged rate — require a hard binary failure call.

The binary call must NOT come from thresholding the disagreement score at the
physical error limit (0.10 m / 3.0°): internal multi-head disagreement is the
spread between heads, not the external pose error, so it does not map 1:1 onto
that limit. Instead we FREEZE an F1-optimal disagreement cutoff on a held-out
synthetic validation split and apply it unchanged to the HIL domains. This is
the same honest train/val protocol our method uses for its operating point
(src/analysis/model_outputs.synth_val_threshold) — same 25% stratified synthetic
hold-out and seed — so baseline and method pick their cutoffs on identical
synthetic val data and never touch HIL labels for tuning.

disagree_* is NaN on PnP-rejected rows (median-imputed upstream), which is
degenerate for this baseline. The HIL baselines are therefore scored on
reject == 0 rows only, and the number of dropped rows is reported.

Our actual method (supervised + importance-weighted classifiers) lives in
src/pipeline/models.py — not here.

Saves:
    results/baseline_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, precision_score, recall_score,
)

PROJECT_ROOT = Path(__file__).parents[2]
RESULTS_DIR  = PROJECT_ROOT / 'results'

DOMAINS = ['lightbox', 'sunlamp']

# axis → error column, absolute fail threshold, and the disagreement score to use
AXES = {
    'translation': {'err_col': 'E_T', 'threshold': 0.10, 'feature': 'disagree_t_m'},
    'rotation':    {'err_col': 'E_R', 'threshold': 3.0,  'feature': 'disagree_R_deg'},
}

# Synthetic val split used to freeze the disagreement cutoff. Matches the model
# operating point (src/analysis/model_outputs.synth_val_threshold): identical
# stratified hold-out fraction and seed, so the baseline and our method select
# their cutoffs on the SAME synthetic val data. HIL is never seen here.
VAL_FRAC     = 0.25
RANDOM_STATE = 42


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


# ── Threshold freezing (synthetic val) ─────────────────────────────────────────
def freeze_disagree_threshold(axis: str, cols_synth: dict, df_synth: pd.DataFrame):
    """Freeze the disagreement cutoff for one axis on a held-out synthetic VAL split.

    The raw disagreement feature is the score (higher = more likely to fail). We
    hold out a stratified VAL_FRAC of synthetic, sweep every candidate cutoff via
    the precision-recall curve, and keep the one that maximizes F1 on that val
    split. HIL is never touched. Returns (threshold, synth_val_f1)."""
    cfg   = AXES[axis]
    ys    = (df_synth[cfg['err_col']].values > cfg['threshold']).astype(int)
    score = cols_synth[cfg['feature']]
    keep  = cols_synth['reject'].astype(int) == 0      # no-op on synthetic; kept for parity
    score, ys = score[keep], ys[keep]

    _, s_val, _, y_val = train_test_split(
        score, ys, test_size=VAL_FRAC, stratify=ys, random_state=RANDOM_STATE)

    # prec/rec have len(thr)+1; align f1 to thr by dropping the trailing point.
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    if not len(thr):
        return np.inf, np.nan
    best = int(np.nanargmax(f1))
    return float(thr[best]), float(f1[best])


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_axis(domain: str, axis: str, cols: dict, df: pd.DataFrame,
                  t_disagree: float, synth_rate: float):
    """Random + disagreement baselines for one (domain, axis), scored on the
    reject == 0 subset. Random flags at the synthetic failure rate; disagreement
    reports threshold-free AUC plus precision / recall / flagged rate from the
    FROZEN synthetic-val cutoff."""
    cfg    = AXES[axis]
    y      = (df[cfg['err_col']].values > cfg['threshold']).astype(int)
    keep   = cols['reject'].astype(int) == 0           # drop degenerate (imputed) rows
    n_drop = int((~keep).sum())

    yk          = y[keep]
    auc_defined = len(np.unique(yk)) > 1               # AUC needs both classes present

    common = {'domain': domain, 'axis': axis,
              'n': int(keep.sum()), 'n_dropped_reject': n_drop,
              'fail_rate': float(yk.mean()) if len(yk) else np.nan}

    # random: flag with probability = SYNTHETIC failure rate. Constant risk score
    # → AUC = 0.5. In expectation, flagged rate = recall = synth_rate and
    # precision = the HIL base rate (independent flagging).
    random_row = {**common, 'baseline': 'random',
                  'auc':          0.5 if auc_defined else np.nan,
                  'precision':    float(yk.mean()) if len(yk) else np.nan,
                  'recall':       synth_rate,
                  'flagged_rate': synth_rate}

    # disagreement: the feature value is the score (higher = more likely to fail).
    # AUC is threshold-free on the continuous score; the binary call uses the
    # frozen synthetic-val cutoff (predict failure when score >= t_disagree), NOT
    # the physical error limit.
    score = cols[cfg['feature']][keep]
    pred  = (score >= t_disagree).astype(int)
    disag_row = {**common, 'baseline': 'disagreement',
                 'auc':          roc_auc_score(yk, score) if auc_defined else np.nan,
                 'threshold':    t_disagree,
                 'precision':    precision_score(yk, pred, zero_division=0),
                 'recall':       recall_score(yk, pred, zero_division=0),
                 'flagged_rate': float(pred.mean()) if len(pred) else np.nan}

    return [random_row, disag_row]


def main():
    # Per-axis synthetic stats (computed once): the random-baseline flag rate and
    # the frozen F1-optimal disagreement cutoff. Both come from synthetic only.
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
                  f"(disagree_* degenerate) → {n_keep} rows, "
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
    print(f"\nSaved → results/baseline_results.csv")


if __name__ == '__main__':
    main()
