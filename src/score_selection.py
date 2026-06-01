# src/score_selection.py
"""
Which score should each component use?  Justification plot for the per-component choice.

For every candidate score — the GMM anomaly score plus each of the 13 model features —
we measure how well it ranks an image's failure-proneness, separately for translation
(E_T) and rotation (E_R), via AUC against the top-30% error label. Some features are
"protective" (higher = safer, e.g. heatmap peak height); for those the discriminative
power is 1-AUC, so we report the EFFECTIVE AUC = max(AUC, 1-AUC) and note the direction.

This makes the per-component score choice in failure_prediction_perfeature.py visible:
the best bar per component is the score worth using.

Label-free w.r.t. HIL: the AUC uses HIL error labels for EVALUATION only — to rank
candidate scores — exactly as we use labels to report metrics, never to fit a predictor.

Saves:
    results/NEW_DATA/score_selection_auc.csv
    figures/score_selection_auc.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'results' / 'NEW_DATA'
FIG_DIR      = PROJECT_ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

DOMAINS    = ['lightbox', 'sunlamp']
COMPONENTS = {'E_T': 'translation', 'E_R': 'rotation'}
PCTILE     = 70.0

# the choice we want to justify (highlighted in the plot)
CHOSEN = {'E_T': 'disagree_t_m', 'E_R': 'gmm'}
DOMAIN_COLORS = {'lightbox': '#4f0942', 'sunlamp': '#d6604d'}


def candidate_scores(domain):
    """{'gmm': ..., <feature name>: ...} for one domain, aligned."""
    gmm   = np.load(DATA_DIR / f'anomaly_scores_gmm_{domain}.npy')
    feats = np.load(DATA_DIR / f'model_features_{domain}.npy')
    names = list(np.load(DATA_DIR / 'model_feature_names.npy'))
    df    = pd.read_csv(DATA_DIR / f'per_image_errors_{domain}.csv')
    n     = min(len(gmm), len(feats), len(df))
    scores = {'gmm': gmm[:n]}
    scores.update({nm: feats[:n, j] for j, nm in enumerate(names)})
    errors = {c: df[c].values[:n] for c in COMPONENTS}
    return scores, errors


def compute_table():
    rows = []
    for dom in DOMAINS:
        scores, errors = candidate_scores(dom)
        for comp in COMPONENTS:
            label = (errors[comp] > np.percentile(errors[comp], PCTILE)).astype(int)
            for name, s in scores.items():
                auc = roc_auc_score(label, s)
                rows.append({'domain': dom, 'component': comp,
                             'comp_name': COMPONENTS[comp], 'score': name,
                             'auc_raw': auc, 'auc_eff': max(auc, 1 - auc),
                             'direction': 'higher=worse' if auc >= 0.5 else 'lower=worse'})
    return pd.DataFrame(rows)


def plot(df):
    fig, axes = plt.subplots(len(DOMAINS), len(COMPONENTS), figsize=(13, 9))
    for r, dom in enumerate(DOMAINS):
        for c, comp in enumerate(COMPONENTS):
            ax = axes[r, c]
            d  = df[(df.domain == dom) & (df.component == comp)].sort_values('auc_eff')
            chosen = CHOSEN[comp]
            colors = [DOMAIN_COLORS[dom] if nm == chosen else '0.7' for nm in d.score]
            ax.barh(d.score, d.auc_eff, color=colors)
            ax.axvline(0.5, ls='--', color='k', lw=1)               # chance
            ax.set_xlim(0.45, max(0.85, d.auc_eff.max() + 0.03))
            ax.set_title(f'{dom} — {COMPONENTS[comp]}   '
                         f'(chosen: {chosen} = {d[d.score==chosen].auc_eff.iloc[0]:.3f})',
                         fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
            if r == len(DOMAINS) - 1:
                ax.set_xlabel('effective AUC  (max(AUC, 1−AUC); 0.5 = chance)')
            ax.grid(alpha=0.25, axis='x')

    fig.suptitle('Which score predicts each failure type best?  '
                 '(highlighted = the score we use for that component)', fontsize=13)
    fig.text(0.5, -0.01,
             'Each bar = one candidate score ranked by AUC against the top-30% error label. '
             'disagree_t_m wins for translation; no feature beats the GMM score for rotation.',
             ha='center', va='top', fontsize=9, style='italic', color='0.3')
    plt.tight_layout()
    out = FIG_DIR / 'score_selection_auc.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → figures/{out.name}')


def main():
    df = compute_table()
    out = DATA_DIR / 'score_selection_auc.csv'
    df.to_csv(out, index=False)

    for dom in DOMAINS:
        for comp in COMPONENTS:
            d = df[(df.domain == dom) & (df.component == comp)].sort_values('auc_eff', ascending=False)
            print(f"\n{dom} / {COMPONENTS[comp]}  — top candidate scores by AUC")
            for _, r in d.head(4).iterrows():
                star = '  ← chosen' if r['score'] == CHOSEN[comp] else ''
                print(f"  {r['score']:18s} AUC={r['auc_eff']:.3f}  ({r['direction']}){star}")

    plot(df)
    print(f"\nSaved → results/NEW_DATA/{out.name}")


if __name__ == '__main__':
    main()
