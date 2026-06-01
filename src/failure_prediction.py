# src/failure_prediction.py
"""
Evaluate the label-free failure predictor on the full HIL domains.

Predictor (deployable, no labels):
    cutoff = 70th percentile of the GMM anomaly scores, calibrated from a small
             unlabeled sample of N_CALIB target images (see threshold_stability.py,
             which showed n≈100 sets a stable cut).
    predict FAILURE  ⇔  anomaly_score > cutoff       (flags the ~top-30% most anomalous)

Ground truth (per domain, per component): an image is a TRUE failure if its SPNv2
error is in the domain's top 30% — i.e. above the 70th percentile of that error
component. Rotation (E_R) and translation (E_T) are judged separately.

Four evaluations:  {lightbox, sunlamp} × {E_R, E_T}.

The predicted set is SHARED across components within a domain (the cut lives on the
anomaly score, which is blind to error type); only the ground truth differs.

Two views of performance:
  • Thresholded metrics (precision/recall/F1/accuracy) at the n=100 cutoff. Because
    the cut is estimated from a finite sample it carries calibration noise, so we
    Monte-Carlo over N_DRAWS independent 100-image samples and report mean ± std.
  • AUC — threshold-free; measures how well the raw score *ranks* failures above
    non-failures, independent of the cutoff. Computed once from the full scores.

Baselines for context: a random 30% flag scores precision = recall = 0.30; AUC = 0.50.

Saves:
    results/NEW_DATA/failure_prediction_gmm.csv
    figures/failure_prediction_roc_gmm.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ── Config ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'results' / 'NEW_DATA'
FIG_DIR      = PROJECT_ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

DETECTOR   = 'gmm'
DOMAINS    = ['lightbox', 'sunlamp']
COMPONENTS = {'E_R': 'rotation', 'E_T': 'translation'}

PCTILE    = 70.0                 # failure = top 30% → cut at the 70th percentile
FAIL_FRAC = 1.0 - PCTILE / 100.  # 0.30 base rate (also the chance precision/recall)
N_CALIB   = 100                  # unlabeled images used to calibrate the cutoff
N_DRAWS   = 1000                 # Monte-Carlo draws of the calibration sample
SEED      = 42

# domain colors matched to the plot_mahal_scores notebook
DOMAIN_COLORS = {'lightbox': '#4f0942', 'sunlamp': '#d6604d'}
COMP_STYLE    = {'E_R': '--', 'E_T': '-'}


# ── Data ────────────────────────────────────────────────────────────────────────
def load_domain(domain: str):
    """Return (scores, {component: raw errors}, {component: top-30% failure mask})."""
    scores = np.load(DATA_DIR / f'anomaly_scores_{DETECTOR}_{domain}.npy')
    df     = pd.read_csv(DATA_DIR / f'per_image_errors_{domain}.csv')
    n      = min(len(scores), len(df))
    scores = scores[:n]
    errors    = {c: df[c].values[:n] for c in COMPONENTS}
    true_fail = {c: errors[c] > np.percentile(errors[c], PCTILE) for c in COMPONENTS}
    return scores, errors, true_fail


# ── Metrics ───────────────────────────────────────────────────────────────────────
def threshold_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Confusion-matrix metrics for one binary prediction vs one truth mask."""
    tp = int(np.count_nonzero(pred & true))
    fp = int(np.count_nonzero(pred & ~true))
    fn = int(np.count_nonzero(~pred & true))
    tn = int(np.count_nonzero(~pred & ~true))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / len(pred)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def evaluate_domain(scores, true_fail, rng):
    """
    AUC (threshold-free) + n=100-calibrated thresholded metrics (mean ± std over
    N_DRAWS calibration samples), for each component of one domain.
    """
    N = len(scores)
    rows = []
    for comp in COMPONENTS:
        truth = true_fail[comp]
        auc = roc_auc_score(truth, scores)   # uses raw scores, no cutoff involved

        # Monte-Carlo the deployable cutoff: calibrate on N_CALIB unlabeled images,
        # apply to ALL images, score against this component's ground truth.
        metric_keys = ('precision', 'recall', 'f1', 'accuracy', 'tp', 'fp', 'fn', 'tn')
        draw = {k: np.empty(N_DRAWS) for k in metric_keys + ('flagged',)}
        for b in range(N_DRAWS):
            cutoff = np.percentile(scores[rng.integers(0, N, size=N_CALIB)], PCTILE)
            pred   = scores > cutoff
            m = threshold_metrics(pred, truth)
            for k in metric_keys:
                draw[k][b] = m[k]
            draw['flagged'][b] = pred.mean()

        row = {'domain': None, 'component': comp, 'comp_name': COMPONENTS[comp],
               'n_total': N, 'auc': auc, 'chance_precision': FAIL_FRAC}
        for k in ('flagged', 'precision', 'recall', 'f1', 'accuracy',
                  'tp', 'fp', 'fn', 'tn'):
            row[f'{k}_mean'] = draw[k].mean()
            row[f'{k}_std']  = draw[k].std()
        # confusion-matrix counts as a % of all images (mean over draws)
        for k in ('tp', 'fp', 'fn', 'tn'):
            row[f'{k}_pct'] = 100.0 * draw[k].mean() / N
        rows.append(row)
    return rows


# ── Plot: error scatter — prediction vs ground truth, side by side ───────────────
GREEN, RED = '#2CA02C', '#D62728'   # green = correct prediction, red = mistake

def _pct_rank(a: np.ndarray) -> np.ndarray:
    """Percentile rank (0–100) of each value — uniform spread, so the plot is readable."""
    order = np.argsort(np.argsort(a))
    return (order + 0.5) / len(a) * 100.0


def plot_error_scatter(scores_by_dom, errors_by_dom):
    """
    Four panels: {lightbox, sunlamp} × {translation, rotation}. Each point = one image at
    (E_T, E_R) on error-PERCENTILE axes (raw errors are too heavy-tailed to read), coloured
    by whether OUR prediction (anomaly score > cutoff) was CORRECT for that component's
    top-30% ground truth. The dashed line marks that component's failure threshold (the 70
    percentile); red on its failure side = misses, red on its pass side = false alarms.
    """
    comp_order = ['E_T', 'E_R']                       # translation, then rotation
    fig, axes = plt.subplots(len(DOMAINS), len(comp_order), figsize=(12, 10))

    for r, dom in enumerate(DOMAINS):
        scores = scores_by_dom[dom]
        eT, eR = errors_by_dom[dom]['E_T'], errors_by_dom[dom]['E_R']
        rT, rR = _pct_rank(eT), _pct_rank(eR)         # axes: error percentile rank
        cutoff = np.percentile(scores, PCTILE)        # ≈ the deployable n=100 cut
        pred   = scores > cutoff

        for c, comp in enumerate(comp_order):
            ax    = axes[r, c]
            truth = errors_by_dom[dom][comp] > np.percentile(errors_by_dom[dom][comp], PCTILE)
            correct = (pred == truth)

            ax.scatter(rT[correct],  rR[correct],  s=7, c=GREEN, alpha=0.30, linewidths=0)
            ax.scatter(rT[~correct], rR[~correct], s=9, c=RED,   alpha=0.55, linewidths=0)

            # this component's failure threshold (vertical for E_T, horizontal for E_R)
            if comp == 'E_T':
                ax.axvspan(PCTILE, 100, color='0.5', alpha=0.06, zorder=0)
                ax.axvline(PCTILE, ls='--', color='0.3', lw=1.2)
            else:
                ax.axhspan(PCTILE, 100, color='0.5', alpha=0.06, zorder=0)
                ax.axhline(PCTILE, ls='--', color='0.3', lw=1.2)

            ax.text(0.97, 0.96, f'accuracy {correct.mean():.2f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            ax.set_xlim(0, 100); ax.set_ylim(0, 100)
            ax.set_title(f'{dom} — {COMPONENTS[comp]}', fontsize=11)
            if r == len(DOMAINS) - 1:
                ax.set_xlabel('translation error percentile')
            if c == 0:
                ax.set_ylabel('rotation error percentile')
            ax.grid(alpha=0.2)

    handles = [plt.Line2D([], [], marker='o', ls='', color=GREEN, label='correct'),
               plt.Line2D([], [], marker='o', ls='', color=RED,   label='mistake')]
    fig.legend(handles=handles, loc='upper right', fontsize=9, ncol=2)
    fig.suptitle('Was our failure prediction correct?   (green = correct, red = mistake)',
                 fontsize=13)
    fig.text(0.5, -0.01,
             'Axes = error percentile rank; dashed line + shading = that component\'s '
             'top-30% failure region. Red on the failure side = missed failures; '
             'red on the pass side = false alarms.',
             ha='center', va='top', fontsize=9, style='italic', color='0.3')
    plt.tight_layout()
    out = FIG_DIR / 'failure_prediction_scatter_gmm.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → figures/{out.name}')


# ── Plot: ROC curves (threshold-free view) ──────────────────────────────────────
def plot_roc(scores_by_dom, truth_by_dom, table):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    auc_lookup = {(r['domain'], r['component']): r['auc'] for r in table}

    for dom in DOMAINS:
        for comp in COMPONENTS:
            fpr, tpr, _ = roc_curve(truth_by_dom[dom][comp], scores_by_dom[dom])
            auc = auc_lookup[(dom, comp)]
            ax.plot(fpr, tpr, COMP_STYLE[comp], color=DOMAIN_COLORS[dom], lw=2,
                    label=f'{dom} / {COMPONENTS[comp]}  (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], ls=':', color='0.5', lw=1, label='chance (AUC=0.50)')
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate (recall)')
    ax.set_title('Failure prediction from GMM anomaly score\nROC per domain × component')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')

    plt.tight_layout()
    out = FIG_DIR / 'failure_prediction_roc_gmm.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → figures/{out.name}')


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    scores_by_dom, errors_by_dom, truth_by_dom, table = {}, {}, {}, []

    for dom in DOMAINS:
        scores, errors, true_fail = load_domain(dom)
        scores_by_dom[dom], errors_by_dom[dom], truth_by_dom[dom] = scores, errors, true_fail
        rows = evaluate_domain(scores, true_fail, rng)
        for r in rows:
            r['domain'] = dom
        table.extend(rows)

    df = pd.DataFrame(table)
    out = DATA_DIR / 'failure_prediction_gmm.csv'
    df.to_csv(out, index=False)

    # ── readable summary ──
    print(f"\n{'='*72}")
    print(f"Label-free failure prediction  (cutoff = 70th pctile of GMM score, "
          f"calibrated from n={N_CALIB})")
    print(f"thresholded metrics: mean ± std over {N_DRAWS} calibration draws;  "
          f"chance precision = {FAIL_FRAC:.2f}, chance AUC = 0.50")
    print('='*72)
    for _, r in df.iterrows():
        print(f"\n{r['domain']} / {r['comp_name']}   (N={int(r['n_total'])} images)")
        print(f"  AUC        {r['auc']:.3f}        (threshold-free; chance 0.500)")
        print(f"  flagged    {r['flagged_mean']:.3f} ± {r['flagged_std']:.3f}   "
              f"(target {FAIL_FRAC:.2f})")
        print(f"  precision  {r['precision_mean']:.3f} ± {r['precision_std']:.3f}   "
              f"(chance {FAIL_FRAC:.2f})")
        print(f"  recall     {r['recall_mean']:.3f} ± {r['recall_std']:.3f}")
        print(f"  F1         {r['f1_mean']:.3f} ± {r['f1_std']:.3f}")
        print(f"  accuracy   {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}")
        # confusion matrix (mean counts over draws, with % of all images)
        print(f"  confusion matrix (mean count, % of all {int(r['n_total'])} images):")
        print(f"                  predicted FAIL          predicted PASS")
        print(f"    true FAIL    TP {r['tp_mean']:7.0f} ({r['tp_pct']:4.1f}%)   "
              f"FN {r['fn_mean']:7.0f} ({r['fn_pct']:4.1f}%)")
        print(f"    true PASS    FP {r['fp_mean']:7.0f} ({r['fp_pct']:4.1f}%)   "
              f"TN {r['tn_mean']:7.0f} ({r['tn_pct']:4.1f}%)")

    plot_roc(scores_by_dom, truth_by_dom, table)
    plot_error_scatter(scores_by_dom, errors_by_dom)
    print(f"\nSaved → results/NEW_DATA/{out.name}")


if __name__ == '__main__':
    main()
