"""
ROC curves in four panels (lightbox / sunlamp x translation / rotation). Each
panel overlays our method (supervised LR), iw, the disagreement baseline, the
HIL-trained oracle upper bound, and the chance diagonal. Uses the per-image scores
from src/analysis/model_outputs.py (the deployed fits).

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from src.analysis._common import (
    plt, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, COMP_NAME, save,
)
from src.analysis.model_outputs import all_domain_outputs
from src.pipeline.models import DOMAINS, COMPONENTS

# panel-title colors only; curve colors use the shared METHOD_COLORS
DOMAIN_COLORS = {'synthetic': '#2166ac', 'lightbox': '#4f0942', 'sunlamp': '#d6604d'}

COMPS = ['E_T', 'E_R']


def _draw_panel(ax, data, dom, comp, show_xlabel, show_ylabel):
    y = data['y_true']
    ax.plot([0, 1], [0, 1], ls=':', color=METHOD_COLORS['random'], lw=1.2,
            label=f"{METHOD_LABELS['random']} (0.50)")
    for name in [m for m in METHOD_ORDER if m != 'random']:
        s  = data['probs'][name]
        ok = np.isfinite(s)
        if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        auc = roc_auc_score(y[ok], s[ok])
        lw  = 2.4 if name == 'method' else 1.6
        ax.plot(fpr, tpr, color=METHOD_COLORS[name], lw=lw,
                label=f'{METHOD_LABELS[name]} ({auc:.2f})')
    ax.set_title(f'{dom.capitalize()} - {COMP_NAME[comp].capitalize()}',
                 fontsize=11, color=DOMAIN_COLORS[dom])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    if show_xlabel:
        ax.set_xlabel('False Positive Rate')
    if show_ylabel:
        ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=7.5, loc='lower right')


def plot(outs):
    # 2x2 grid: domains down rows, components across columns
    fig, axes = plt.subplots(len(DOMAINS), len(COMPS), figsize=(7.5, 7))
    for r, dom in enumerate(DOMAINS):
        for c, comp in enumerate(COMPS):
            _draw_panel(axes[r, c], outs[dom][comp], dom, comp,
                        show_xlabel=(r == len(DOMAINS) - 1), show_ylabel=(c == 0))
    fig.suptitle('Failure-Prediction ROC by Domain x Axis  (AUC in legend)',
                 fontweight='bold')
    fig.tight_layout()
    save(fig, 'roc_curves')


def plot_row(outs):
    # single row: all four domain x component panels side by side
    groups = [(d, c) for d in DOMAINS for c in COMPS]
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 4))
    for i, (dom, comp) in enumerate(groups):
        _draw_panel(axes[i], outs[dom][comp], dom, comp,
                    show_xlabel=True, show_ylabel=(i == 0))
    fig.suptitle('Failure-Prediction ROC by Domain x Axis  (AUC in legend)',
                 fontweight='bold')
    fig.tight_layout()
    save(fig, 'roc_curves_row')


def main():
    outs = all_domain_outputs()
    plot(outs)
    plot_row(outs)


if __name__ == '__main__':
    main()
