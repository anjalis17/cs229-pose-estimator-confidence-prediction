# src/analysis/plot_auc_bars.py
"""
Beat 3 — AUC summary bar chart.

Method vs importance-weighted vs oracle AUC, grouped by domain × axis, with the
random chance line at 0.50. Shows (a) the method works, (b) IW barely moves it,
(c) how much headroom remains up to the HIL-trained oracle.

Uses the per-image scores from src/analysis/model_outputs.py.

Saves:
    figures/auc_bars.{png,pdf}
    results/analysis_auc.csv
"""

# allow running directly: put repo root on sys.path so `import src` resolves
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
from src.pipeline.models import DOMAINS, COMPONENTS

TABLE_METHODS = ['method', 'iw', 'disagreement', 'oracle']   # saved to CSV
BARS          = ['method', 'iw', 'disagreement', 'oracle']   # subset drawn as bars


def _auc_table(outs):
    rows = []
    for dom in DOMAINS:
        for comp in COMPONENTS:
            data = outs[dom][comp]
            y = data['y_true']
            for name in TABLE_METHODS:
                s  = data['probs'][name]
                ok = np.isfinite(s)
                auc = (roc_auc_score(y[ok], s[ok])
                       if ok.sum() and len(np.unique(y[ok])) > 1 else np.nan)
                rows.append({'domain': dom, 'component': comp,
                             'method': name, 'auc': auc})
    return pd.DataFrame(rows)


def plot(df):
    groups = [(d, c) for d in DOMAINS for c in COMPONENTS]
    x = np.arange(len(groups))
    n = len(BARS)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, name in enumerate(BARS):
        vals = [df[(df.domain == d) & (df.component == c) & (df.method == name)]
                .auc.iloc[0] for d, c in groups]
        bars = ax.bar(x + (k - (n - 1) / 2) * w, vals, width=w,
                      color=METHOD_COLORS[name], label=METHOD_LABELS[name])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=7.5)
    ax.axhline(0.5, color=METHOD_COLORS['random'], ls='--', lw=1.2, label='random (0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}\n{COMP_NAME[c]}' for d, c in groups])
    ax.set_ylabel('AUC')
    ax.set_ylim(0.45, 1.0)
    ax.set_title('Failure-prediction AUC: method vs importance-weighted vs oracle',
                 fontweight='bold')
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
    print(f'Saved → results/{out.name}')
    plot(df)


if __name__ == '__main__':
    main()
