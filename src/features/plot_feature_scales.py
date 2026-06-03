"""
Preprocessing justification figure: per-feature standard deviation on the
synthetic split, on a log x-axis.

Motivates two decisions at once:
  1. drop 'reject'  - constant (std=0) on synthetic, no in-distribution signal,
     and it breaks scale-based methods / the Mahalanobis ridge.
  2. standardize the rest - the kept features span ~7 orders of magnitude
     (bbox_area ~1e4 vs seg_entropy ~1e-3), so any distance- or penalty-based
     model is otherwise dominated by the large-scale features.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'
FIG_DIR = PROJECT_ROOT / 'figures'

DROP_FEATURES = ['reject']


def main():
    names = np.load(RESULTS_DIR / 'model_feature_names.npy', allow_pickle=True).tolist()
    X = np.load(RESULTS_DIR / 'model_features_synthetic.npy')
    std = X.std(axis=0)

    kept = [(n, s) for n, s in zip(names, std) if n not in DROP_FEATURES]
    dropped = [(n, s) for n, s in zip(names, std) if n in DROP_FEATURES]

    kept.sort(key=lambda t: t[1])
    kept_names = [n for n, _ in kept]
    kept_std = np.array([s for _, s in kept])

    smin, smax = kept_std.min(), kept_std.max()
    n_orders = np.log10(smax / smin)

    plt.rcdefaults()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    floor = smin / 50.0   # log-axis can't show std=0, so park 'reject' here
    y_kept = np.arange(len(dropped), len(dropped) + len(kept_names))

    ax.barh(y_kept, kept_std, color='#4C72B0', edgecolor='white',
            height=0.7, label='kept (12 features)')
    for (dn, _), y in zip(dropped, range(len(dropped))):
        ax.barh(y, smax, left=floor, color='none', edgecolor='#e74c3c',
                hatch='///', height=0.7,
                label='dropped: constant on synthetic (std = 0)')
        ax.text(floor, y, '  std = 0  ->  dropped', va='center', ha='left',
                fontsize=8, color='#e74c3c', fontweight='bold')

    ax.set_xscale('log')
    ax.set_yticks(list(range(len(dropped))) + list(y_kept))
    ax.set_yticklabels([dn for dn, _ in dropped] + kept_names, fontsize=8.5)
    ax.set_xlabel('Standard deviation on synthetic split  (log scale)')
    ax.set_title('Feature scales motivate dropping `reject` and standardizing',
                 fontsize=10, pad=10)

    ax.axvline(smin, color='0.6', ls='--', lw=0.8)
    ax.axvline(smax, color='0.6', ls='--', lw=0.8)
    ymid = len(dropped) + len(kept_names) * 0.5
    ax.annotate('', xy=(smax, ymid), xytext=(smin, ymid),
                arrowprops=dict(arrowstyle='<->', color='0.35', lw=1.3))
    ax.text(np.sqrt(smin * smax), ymid + 0.4,
            f'~{n_orders:.0f} orders of magnitude\n'
            r'-> standardize ($\mu$=0, $\sigma$=1)',
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
    print(f'Saved -> figures/feature_preprocessing.(png|pdf)')


if __name__ == '__main__':
    main()
