# src/analysis/plot_feature_correlation.py
"""
Beat 2 — our features see the gap.

Per-feature Spearman ρ between each SPNv2 model-internal feature and the pose
error, computed separately for translation (E_T) and rotation (E_R) and for each
domain. Features whose |ρ| is large and consistent in sign across domains are the
ones that carry failure signal through the domain gap.

Reads the saved feature matrices + per-image errors — no model needed.

Saves:
    figures/feature_spearman.{png,pdf}
"""

# allow running directly: put repo root on sys.path so `import src` resolves
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis._common import plt, RESULTS_DIR, DOMAIN_COLORS, COMP_NAME, save

DOMAINS    = ['synthetic', 'lightbox', 'sunlamp']
COMPONENTS = ['E_T', 'E_R']


def _feature_table():
    """Long table: domain × feature × component → Spearman ρ."""
    names = [str(n) for n in np.load(RESULTS_DIR / 'model_feature_names.npy',
                                     allow_pickle=True)]
    rows = []
    for d in DOMAINS:
        X  = np.load(RESULTS_DIR / f'model_features_{d}.npy')
        df = pd.read_csv(RESULTS_DIR / f'per_image_errors_{d}.csv')
        n  = min(len(X), len(df))
        X, df = X[:n], df.iloc[:n]
        for comp in COMPONENTS:
            for j, nm in enumerate(names):
                col = X[:, j]
                rho = np.nan if np.nanstd(col) == 0 else spearmanr(col, df[comp].values,
                                                                   nan_policy='omit').statistic
                rows.append({'domain': d, 'feature': nm, 'component': comp, 'rho': rho})
    return pd.DataFrame(rows), names


def plot(df, names):
    # order features by mean |ρ| over the HIL domains (translation) — most useful at top
    hil = df[(df.domain != 'synthetic')]
    order = (hil.assign(absrho=hil.rho.abs())
                .groupby('feature').absrho.mean().sort_values().index.tolist())
    y = np.arange(len(order))
    bar_h = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(order) + 2), sharey=True)
    for ax, comp in zip(axes, COMPONENTS):
        for k, d in enumerate(DOMAINS):
            sub = df[(df.component == comp) & (df.domain == d)].set_index('feature')
            vals = [sub.loc[f, 'rho'] if f in sub.index else np.nan for f in order]
            ax.barh(y + (k - 1) * bar_h, vals, height=bar_h,
                    color=DOMAIN_COLORS[d], label=d)
        ax.axvline(0, color='k', lw=0.8)
        for x in (-0.3, 0.3):
            ax.axvline(x, color='0.6', ls=':', lw=1)   # |ρ|>0.3 "useful" guide
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel('Spearman ρ')
        ax.set_title(f'{COMP_NAME[comp].capitalize()} error  ($E_{comp[-1]}$)')
        ax.set_xlim(-1, 1)
        ax.grid(alpha=0.2, axis='x')
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    fig.suptitle('Per-feature Spearman ρ vs pose error  '
                 '(dotted = |ρ| = 0.3 usefulness guide)', fontweight='bold')
    fig.tight_layout()
    save(fig, 'feature_spearman')


def main():
    df, names = _feature_table()
    wide = (df.pivot_table(index='feature', columns=['component', 'domain'], values='rho')
              .reindex(names))
    wide.to_csv(RESULTS_DIR / 'feature_spearman.csv')
    print(wide.to_string(float_format=lambda v: f'{v:.3f}'))
    print('Saved → results/feature_spearman.csv')
    plot(df, names)


if __name__ == '__main__':
    main()
