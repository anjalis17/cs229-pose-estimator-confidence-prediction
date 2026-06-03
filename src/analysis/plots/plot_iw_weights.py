"""
Histogram of the synthetic importance weights w = p/(1-p) toward each HIL domain,
where p = P(HIL|x) from the domain classifier. A spike near w=0 with a thin tail
means most synthetic images look nothing like the target, so reweighting
concentrates on a few samples (low effective sample size) and barely moves the
failure classifier, consistent with IW ~ supervised in the ROC/AUC. Reuses the
domain classifier + weighting from src/pipeline/models.py.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis._common import plt, RESULTS_DIR, DOMAIN_COLORS, save
from src.pipeline.models import (
    load_domain, domain_probabilities, importance_weights, DOMAINS, WEIGHT_CLIP_PCT,
)


def _ess_fraction(w):
    # Kish effective sample size as a fraction of n: (sum w)^2 / (n * sum w^2)
    w = np.asarray(w, float)
    return (w.sum() ** 2) / (len(w) * (w ** 2).sum())


def main():
    X_synth, _ = load_domain('synthetic')
    stats = []

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(5.2 * len(DOMAINS), 3.8),
                             squeeze=False)
    for ax, dom in zip(axes[0], DOMAINS):
        X_hil, _ = load_domain(dom)
        p_oof, yd = domain_probabilities(X_synth, X_hil)
        w, cap = importance_weights(p_oof[:len(X_synth)])
        ess = _ess_fraction(w)
        domain_auc = roc_auc_score(yd, p_oof)
        stats.append({'domain': dom, 'domain_classifier_auc': domain_auc,
                      'weight_mean': w.mean(), 'weight_max': w.max(),
                      'ess_fraction': ess, 'clip_cap': cap})

        ax.hist(w, bins=60, color=DOMAIN_COLORS[dom], alpha=0.85, edgecolor='none')
        ax.axvline(1.0, color='0.35', ls='--', lw=1.2, label='w = 1 (no reweighting)')
        ax.set_yscale('log')
        ax.set_xlabel('importance weight  w = p/(1-p)')
        ax.set_ylabel('count (log)')
        ax.set_title(f'{dom}\nmean={w.mean():.2f}, max={w.max():.2f}, '
                     f'ESS={ess:.1%}  (clip p{WEIGHT_CLIP_PCT}={cap:.1f})', fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(alpha=0.2)
    fig.suptitle('Synthetic importance weights toward each HIL domain  '
                 '(low ESS -> IW barely moves the classifier)', fontweight='bold')
    fig.tight_layout()

    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(RESULTS_DIR / 'iw_weight_stats.csv', index=False)
    print(df_stats.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('Saved -> results/iw_weight_stats.csv')
    save(fig, 'iw_weights')


if __name__ == '__main__':
    main()
