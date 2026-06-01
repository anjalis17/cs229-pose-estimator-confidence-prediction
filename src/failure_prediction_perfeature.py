# src/failure_prediction_perfeature.py
"""
Per-component failure prediction — a variant of failure_prediction.py that lets each
error component use its OWN best label-free score instead of a single shared GMM score.

Motivation (from the per-feature AUC sweep): translation failures are predicted better
by the pose heads' translation disagreement (`disagree_t_m`, AUC ≈0.78–0.82) than by the
GMM anomaly score (≈0.73–0.77), while rotation has no stronger single feature so it keeps
the GMM score. So we run two independent pipelines with different scores:

    SCORE_SOURCE = {'E_T': 'disagree_t_m',   # translation → translation-head disagreement
                    'E_R': 'gmm'}            # rotation    → GMM anomaly score

Everything else is identical to failure_prediction.py and stays rank-based: per component
we calibrate a 70th-percentile cutoff from N_CALIB unlabeled target images, flag scores
above it, and score against that component's top-30% error ground truth. No HIL labels are
used to fit anything. This file does NOT modify the GMM-only pipeline; compare the two CSVs.

Scores:
    'gmm'        → results/NEW_DATA/anomaly_scores_gmm_{domain}.npy
    any other    → a named column of results/NEW_DATA/model_features_{domain}.npy
                   (higher value must mean "more likely to fail")

Saves:
    results/NEW_DATA/failure_prediction_perfeature.csv
    figures/failure_prediction_perfeature_roc.png
    figures/failure_prediction_perfeature_scatter.png
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

DOMAINS    = ['lightbox', 'sunlamp']
COMPONENTS = {'E_R': 'rotation', 'E_T': 'translation'}

# the per-component score choice — edit here to try other features
SCORE_SOURCE = {'E_T': 'disagree_t_m', 'E_R': 'gmm'}

PCTILE    = 70.0
FAIL_FRAC = 1.0 - PCTILE / 100.
N_CALIB   = 100
N_DRAWS   = 1000
SEED      = 42

DOMAIN_COLORS = {'lightbox': '#4f0942', 'sunlamp': '#d6604d'}
COMP_STYLE    = {'E_R': '--', 'E_T': '-'}
GREEN, RED    = '#2CA02C', '#D62728'


# ── Data ────────────────────────────────────────────────────────────────────────
def load_domain(domain: str):
    """
    Returns:
      scores  {component: (N,) score chosen by SCORE_SOURCE, higher = more likely fail}
      errors  {component: (N,) raw error}
      truth   {component: (N,) top-30% failure mask}
    All aligned to a common length.
    """
    gmm   = np.load(DATA_DIR / f'anomaly_scores_gmm_{domain}.npy')
    feats = np.load(DATA_DIR / f'model_features_{domain}.npy')
    names = list(np.load(DATA_DIR / 'model_feature_names.npy'))
    df    = pd.read_csv(DATA_DIR / f'per_image_errors_{domain}.csv')
    n     = min(len(gmm), len(feats), len(df))

    def score_for(src):
        if src == 'gmm':
            return gmm[:n]
        if src not in names:
            raise KeyError(f"'{src}' not in model_feature_names: {names}")
        return feats[:n, names.index(src)]

    scores = {comp: score_for(SCORE_SOURCE[comp]) for comp in COMPONENTS}
    errors = {comp: df[comp].values[:n] for comp in COMPONENTS}
    truth  = {comp: errors[comp] > np.percentile(errors[comp], PCTILE) for comp in COMPONENTS}
    return scores, errors, truth


# ── Metrics ───────────────────────────────────────────────────────────────────────
def threshold_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
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


def evaluate_domain(scores, truth, rng):
    """AUC + n=100-calibrated thresholded metrics (mean ± std over draws), per component."""
    rows = []
    metric_keys = ('precision', 'recall', 'f1', 'accuracy', 'tp', 'fp', 'fn', 'tn')
    for comp in COMPONENTS:
        s, t = scores[comp], truth[comp]
        N = len(s)
        auc = roc_auc_score(t, s)

        draw = {k: np.empty(N_DRAWS) for k in metric_keys + ('flagged',)}
        for b in range(N_DRAWS):
            cutoff = np.percentile(s[rng.integers(0, N, size=N_CALIB)], PCTILE)
            pred   = s > cutoff
            m = threshold_metrics(pred, t)
            for k in metric_keys:
                draw[k][b] = m[k]
            draw['flagged'][b] = pred.mean()

        row = {'domain': None, 'component': comp, 'comp_name': COMPONENTS[comp],
               'score': SCORE_SOURCE[comp], 'n_total': N, 'auc': auc,
               'chance_precision': FAIL_FRAC}
        for k in ('flagged',) + metric_keys:
            row[f'{k}_mean'] = draw[k].mean()
            row[f'{k}_std']  = draw[k].std()
        for k in ('tp', 'fp', 'fn', 'tn'):
            row[f'{k}_pct'] = 100.0 * draw[k].mean() / N
        rows.append(row)
    return rows


# ── Plots ─────────────────────────────────────────────────────────────────────────
def plot_roc(scores_by_dom, truth_by_dom, table):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    auc_lookup = {(r['domain'], r['component']): r['auc'] for r in table}
    for dom in DOMAINS:
        for comp in COMPONENTS:
            fpr, tpr, _ = roc_curve(truth_by_dom[dom][comp], scores_by_dom[dom][comp])
            auc = auc_lookup[(dom, comp)]
            ax.plot(fpr, tpr, COMP_STYLE[comp], color=DOMAIN_COLORS[dom], lw=2,
                    label=f'{dom} / {COMPONENTS[comp]}  [{SCORE_SOURCE[comp]}]  (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], ls=':', color='0.5', lw=1, label='chance (AUC=0.50)')
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate (recall)')
    ax.set_title('Per-component failure prediction\nROC (each component uses its own score)')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    plt.tight_layout()
    out = FIG_DIR / 'failure_prediction_perfeature_roc.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → figures/{out.name}')


def _pct_rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(a))
    return (order + 0.5) / len(a) * 100.0


def plot_error_scatter(scores_by_dom, errors_by_dom):
    """Four correctness panels (domain × component); each panel uses its component's score."""
    comp_order = ['E_T', 'E_R']
    fig, axes = plt.subplots(len(DOMAINS), len(comp_order), figsize=(12, 10))
    for r, dom in enumerate(DOMAINS):
        eT, eR = errors_by_dom[dom]['E_T'], errors_by_dom[dom]['E_R']
        rT, rR = _pct_rank(eT), _pct_rank(eR)
        for c, comp in enumerate(comp_order):
            ax    = axes[r, c]
            s     = scores_by_dom[dom][comp]
            cutoff = np.percentile(s, PCTILE)          # ≈ deployable n=100 cut
            pred  = s > cutoff
            truth = errors_by_dom[dom][comp] > np.percentile(errors_by_dom[dom][comp], PCTILE)
            correct = (pred == truth)

            ax.scatter(rT[correct],  rR[correct],  s=7, c=GREEN, alpha=0.30, linewidths=0)
            ax.scatter(rT[~correct], rR[~correct], s=9, c=RED,   alpha=0.55, linewidths=0)
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
            ax.set_title(f'{dom} — {COMPONENTS[comp]}  [{SCORE_SOURCE[comp]}]', fontsize=10)
            if r == len(DOMAINS) - 1:
                ax.set_xlabel('translation error percentile')
            if c == 0:
                ax.set_ylabel('rotation error percentile')
            ax.grid(alpha=0.2)

    handles = [plt.Line2D([], [], marker='o', ls='', color=GREEN, label='correct'),
               plt.Line2D([], [], marker='o', ls='', color=RED,   label='mistake')]
    fig.legend(handles=handles, loc='upper right', fontsize=9, ncol=2)
    fig.suptitle('Per-component failure prediction — was it correct?   '
                 '(green = correct, red = mistake)', fontsize=13)
    fig.text(0.5, -0.01,
             'Each panel uses that component\'s own score (shown in title). Axes = error '
             'percentile rank; shaded = top-30% failure region.',
             ha='center', va='top', fontsize=9, style='italic', color='0.3')
    plt.tight_layout()
    out = FIG_DIR / 'failure_prediction_perfeature_scatter.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → figures/{out.name}')


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    scores_by_dom, errors_by_dom, truth_by_dom, table = {}, {}, {}, []

    for dom in DOMAINS:
        scores, errors, truth = load_domain(dom)
        scores_by_dom[dom], errors_by_dom[dom], truth_by_dom[dom] = scores, errors, truth
        rows = evaluate_domain(scores, truth, rng)
        for r in rows:
            r['domain'] = dom
        table.extend(rows)

    df = pd.DataFrame(table)
    out = DATA_DIR / 'failure_prediction_perfeature.csv'
    df.to_csv(out, index=False)

    print(f"\n{'='*72}")
    print("Per-component label-free failure prediction")
    print(f"scores: {SCORE_SOURCE}   (cutoff = 70th pctile, calibrated from n={N_CALIB})")
    print(f"thresholded metrics: mean ± std over {N_DRAWS} draws;  "
          f"chance precision = {FAIL_FRAC:.2f}, chance AUC = 0.50")
    print('='*72)
    for _, r in df.iterrows():
        print(f"\n{r['domain']} / {r['comp_name']}   "
              f"[score: {r['score']}]   (N={int(r['n_total'])})")
        print(f"  AUC        {r['auc']:.3f}        (threshold-free; chance 0.500)")
        print(f"  precision  {r['precision_mean']:.3f} ± {r['precision_std']:.3f}   "
              f"(chance {FAIL_FRAC:.2f})")
        print(f"  recall     {r['recall_mean']:.3f} ± {r['recall_std']:.3f}")
        print(f"  F1         {r['f1_mean']:.3f} ± {r['f1_std']:.3f}")
        print(f"  accuracy   {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}")
        print(f"  confusion (mean count, % of {int(r['n_total'])}):")
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
