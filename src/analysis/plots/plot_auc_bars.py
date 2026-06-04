"""
AUC summary bar chart: method vs importance-weighted vs disagreement vs oracle,
grouped by domain x axis, with a random-chance line at 0.50 and bootstrap 95% CIs.
Shows the method works, IW barely moves it, and the headroom up to the
HIL-trained oracle. Uses the per-image scores from src/analysis/model_outputs.py.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis._common import (
    plt, RESULTS_DIR, METHOD_COLORS, METHOD_LABELS, COMP_NAME, save,
)
from src.analysis.model_outputs import all_domain_outputs
from src.pipeline.models import DOMAINS, COMPONENTS, RANDOM_STATE

# notebook palette → domain tick-label colors (bar colors keep METHOD_COLORS)
DOMAIN_COLORS = {'synthetic': '#2166ac', 'lightbox': '#4f0942', 'sunlamp': '#d6604d'}
METHOD_LABELS = {**METHOD_LABELS, 'disagreement': 'disagreement feature'}

TABLE_METHODS = ['method', 'iw', 'disagreement', 'oracle']
BARS = ['method', 'iw', 'disagreement', 'oracle']

N_BOOT = 2000  # bootstrap resamples for the AUC confidence interval
_RNG = np.random.default_rng(RANDOM_STATE)


def _bootstrap_auc_ci(y, s, n_boot=N_BOOT):
    # point AUC + 95% CI by resampling the eval set with replacement; the CI lets
    # near-equal bars (and apparent 'beating' of the oracle) read as ties
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, s)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = _RNG.integers(0, n, n)
        if len(np.unique(y[idx])) > 1:
            boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi


def _auc_table(outs):
    rows = []
    for dom in DOMAINS:
        for comp in COMPONENTS:
            data = outs[dom][comp]
            y = data['y_true']
            for name in TABLE_METHODS:
                auc, lo, hi = _bootstrap_auc_ci(y, data['probs'][name])
                rows.append({'domain': dom, 'component': comp, 'method': name,
                             'auc': auc, 'ci_lo': lo, 'ci_hi': hi})
    return pd.DataFrame(rows)


def plot(df):
    groups = [(d, c) for d in DOMAINS for c in COMPONENTS]
    x = np.arange(len(groups))
    n = len(BARS)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, name in enumerate(BARS):
        sub = [df[(df.domain == d) & (df.component == c) & (df.method == name)].iloc[0]
               for d, c in groups]
        vals = np.array([r.auc for r in sub])
        # asymmetric error bars from the bootstrap CI (clip tiny negatives from rounding)
        yerr = np.array([np.clip(vals - [r.ci_lo for r in sub], 0, None),
                         np.clip([r.ci_hi for r in sub] - vals, 0, None)])
        bars = ax.bar(x + (k - (n - 1) / 2) * w, vals, width=w,
                      color=METHOD_COLORS[name], label=METHOD_LABELS[name],
                      yerr=yerr, capsize=2.5,
                      error_kw=dict(lw=0.9, ecolor='#333333'))
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=7.5)
    ax.axhline(0.5, color=METHOD_COLORS['random'], ls='--', lw=1.2, label='random (0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}\n{COMP_NAME[c]}' for d, c in groups])
    for tick, (d, _) in zip(ax.get_xticklabels(), groups):
        tick.set_color(DOMAIN_COLORS[d])      # color domain labels to the notebook palette
    ax.set_ylabel('AUC')
    ax.set_ylim(0.45, 1.0)
    ax.set_title('Failure-Prediction AUC Across Methods', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, ncol=5, loc='upper center',
              bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    save(fig, 'auc_bars')


def main():
    df = _auc_table(all_domain_outputs())
    out = RESULTS_DIR / 'analysis_auc.csv'
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'Saved -> results/{out.name}')
    plot(df)


if __name__ == '__main__':
    main()
