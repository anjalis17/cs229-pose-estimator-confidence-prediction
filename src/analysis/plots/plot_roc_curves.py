# src/analysis/plot_roc_curves.py
"""
Beat 3 — our method catches failures, the dumb baselines don't.

ROC curves in four panels (lightbox / sunlamp × translation / rotation). Each
panel overlays:
    method        supervised LR (ours)
    iw            importance-weighted LR (ours)
    disagreement  disagreement-threshold baseline
    oracle        HIL-trained upper bound
    random        chance diagonal

Uses the per-image scores from src/analysis/model_outputs.py (the deployed fits).

Saves:
    figures/roc_curves.{png,pdf}
"""

# allow running directly: put repo root on sys.path so `import src` resolves
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from src.analysis._common import (
    plt, DOMAIN_COLORS, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER,
    COMP_NAME, save,
)
from src.analysis.model_outputs import all_domain_outputs
from src.pipeline.models import DOMAINS, COMPONENTS


def plot(outs):
    comps = ['E_T', 'E_R']
    fig, axes = plt.subplots(len(DOMAINS), len(comps), figsize=(11, 10))
    for r, dom in enumerate(DOMAINS):
        for c, comp in enumerate(comps):
            ax   = axes[r, c]
            data = outs[dom][comp]
            y    = data['y_true']
            ax.plot([0, 1], [0, 1], ls=':', color=METHOD_COLORS['random'], lw=1.2,
                    label=f"{METHOD_LABELS['random']} (0.50)")
            for name in [m for m in METHOD_ORDER if m != 'random']:
                s = data['probs'][name]
                ok = np.isfinite(s)
                if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y[ok], s[ok])
                auc = roc_auc_score(y[ok], s[ok])
                lw = 2.4 if name == 'method' else 1.6
                ax.plot(fpr, tpr, color=METHOD_COLORS[name], lw=lw,
                        label=f'{METHOD_LABELS[name]} ({auc:.2f})')
            ax.set_title(f'{dom} — {COMP_NAME[comp]}', fontsize=11,
                         color=DOMAIN_COLORS[dom])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.grid(alpha=0.25)
            if r == len(DOMAINS) - 1:
                ax.set_xlabel('false positive rate')
            if c == 0:
                ax.set_ylabel('true positive rate')
            ax.legend(fontsize=7.5, loc='lower right')
    fig.suptitle('Failure-prediction ROC by domain × axis  (AUC in legend)',
                 fontweight='bold')
    fig.tight_layout()
    save(fig, 'roc_curves')


def main():
    plot(all_domain_outputs())


if __name__ == '__main__':
    main()
