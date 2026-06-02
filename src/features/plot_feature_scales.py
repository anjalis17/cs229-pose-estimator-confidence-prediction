# src/plot_feature_scales.py
"""
Preprocessing justification figure.

One horizontal bar chart of each feature's standard deviation on the synthetic
(in-distribution) split, on a log x-axis. It motivates two preprocessing
decisions at once:

  1. DROP 'reject'  — it is constant (std = 0) on synthetic, so it carries no
     in-distribution signal and breaks scale-based methods (and the Mahalanobis
     ridge). It cannot even be placed on a log axis, so it is flagged separately.

  2. STANDARDIZE the rest — the kept features span ~7 orders of magnitude
     (bbox_area ~1e4 vs seg_entropy ~1e-3). Without standardization, any
     distance- or penalty-based model (OC-SVM RBF, logistic-regression L2) is
     dominated by the few large-scale features and ignores the rest.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
RESULTS_DIR  = PROJECT_ROOT / 'results'
FIG_DIR      = PROJECT_ROOT / 'figures'

DROP_FEATURES = ['reject']


def main():
    names = np.load(RESULTS_DIR / 'model_feature_names.npy', allow_pickle=True).tolist()
    X = np.load(RESULTS_DIR / 'model_features_synthetic.npy')
    std = X.std(axis=0)

    # split into kept vs dropped
    kept = [(n, s) for n, s in zip(names, std) if n not in DROP_FEATURES]
    dropped = [(n, s) for n, s in zip(names, std) if n in DROP_FEATURES]

    # sort kept ascending so the dynamic range reads bottom→top
    kept.sort(key=lambda t: t[1])
    kept_names = [n for n, _ in kept]
    kept_std   = np.array([s for _, s in kept])

    smin, smax = kept_std.min(), kept_std.max()
    n_orders = np.log10(smax / smin)

    # ---- plot -------------------------------------------------------------
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # a log-axis floor for the constant 'reject' bar so it is visible & labelled
    floor = smin / 50.0
    y_kept = np.arange(len(dropped), len(dropped) + len(kept_names))

    ax.barh(y_kept, kept_std, color='#4C72B0', edgecolor='white',
            height=0.7, label='kept (12 features)')
    for (dn, _), y in zip(dropped, range(len(dropped))):
        ax.barh(y, smax, left=floor, color='none', edgecolor='#e74c3c',
                hatch='///', height=0.7,
                label='dropped: constant on synthetic (std = 0)')
        ax.text(floor, y, '  std = 0  →  dropped', va='center', ha='left',
                fontsize=8, color='#e74c3c', fontweight='bold')

    ax.set_xscale('log')
    ax.set_yticks(list(range(len(dropped))) + list(y_kept))
    ax.set_yticklabels([dn for dn, _ in dropped] + kept_names, fontsize=8.5)
    ax.set_xlabel('Standard deviation on synthetic split  (log scale)')
    ax.set_title('Feature scales motivate dropping `reject` and standardizing',
                 fontsize=10, pad=10)

    # annotate the dynamic range across the kept features
    ax.axvline(smin, color='0.6', ls='--', lw=0.8)
    ax.axvline(smax, color='0.6', ls='--', lw=0.8)
    ymid = len(dropped) + len(kept_names) * 0.5
    ax.annotate('', xy=(smax, ymid), xytext=(smin, ymid),
                arrowprops=dict(arrowstyle='<->', color='0.35', lw=1.3))
    ax.text(np.sqrt(smin * smax), ymid + 0.4,
            f'~{n_orders:.0f} orders of magnitude\n→ standardize (μ=0, σ=1)',
            ha='center', va='bottom', fontsize=8.5, color='0.2',
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff7e6', ec='0.7'))

    ax.legend(loc='lower right', fontsize=8, frameon=True)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / 'feature_preprocessing'
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    print(f'min std = {smin:.4g} ({kept_names[0]}),  max std = {smax:.4g} ({kept_names[-1]})')
    print(f'dynamic range = {smax/smin:.3g}x  (~{n_orders:.1f} orders of magnitude)')
    print(f'Saved → figures/feature_preprocessing.(png|pdf)')


if __name__ == '__main__':
    main()
