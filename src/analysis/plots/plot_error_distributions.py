# src/analysis/plot_error_distributions.py
"""
Beat 1 — there IS a domain gap.

Plot 1: pose-error (SPEED score) distribution per domain.
Plot 2: the same error split into its translation (E_T) and rotation (E_R)
        components, with the absolute failure thresholds marked.

Both read only the per-image error CSVs — no model needed.

Saves:
    figures/pose_error_distribution.{png,pdf}
    figures/pose_error_components.{png,pdf}
"""

# allow running directly: put repo root on sys.path so `import src` resolves
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis._common import (
    plt, RESULTS_DIR, DOMAIN_COLORS, COMP_NAME, COMP_UNIT, COMP_THRESHOLD, save,
)

DOMAINS = ['synthetic', 'lightbox', 'sunlamp']


def _load():
    # full per-image error splits (same source as the notebook), not the old
    # 1001-row results/per_image_errors_*.csv
    return {d: pd.read_csv(f'per_image_errors_{d}.csv') for d in DOMAINS}


def error_stats(errs):
    """Per-domain error summary + absolute-threshold failure rates → CSV."""
    rows = []
    for d in DOMAINS:
        df  = errs[d]
        row = {'domain': d, 'n': len(df)}
        for col in ('E_T', 'E_R', 'speed_score'):
            v = df[col].values
            row[f'{col}_mean']   = v.mean()
            row[f'{col}_median'] = np.median(v)
            row[f'{col}_p90']    = np.percentile(v, 90)
            row[f'{col}_p99']    = np.percentile(v, 99)
        row['fail_rate_E_T'] = (df['E_T'] > COMP_THRESHOLD['E_T']).mean()
        row['fail_rate_E_R'] = (df['E_R'] > COMP_THRESHOLD['E_R']).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / 'error_distribution_stats.csv', index=False)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('Saved → results/error_distribution_stats.csv')


def _log_hist(ax, values, color, label, n_bins=60):
    """Density histogram on a log x-axis (pose errors are strictly positive,
    heavy-tailed)."""
    v = np.asarray(values, float)
    v = v[v > 0]
    bins = np.logspace(np.log10(v.min()), np.log10(np.percentile(v, 99.5)), n_bins)
    ax.hist(v, bins=bins, density=True, histtype='stepfilled',
            color=color, alpha=0.45, label=label)
    ax.hist(v, bins=bins, density=True, histtype='step', color=color, lw=1.6)
    ax.set_xscale('log')


def plot_speed_score(errs):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for d in DOMAINS:
        med = errs[d]['speed_score'].median()
        _log_hist(ax, errs[d]['speed_score'], DOMAIN_COLORS[d],
                  f'{d}  (median={med:.3f})')
        ax.axvline(med, color=DOMAIN_COLORS[d], ls='--', lw=1.4)
    ax.set_xlabel('SPEED score  =  E_R [rad] + E_T / ||t||   (log scale)')
    ax.set_ylabel('density')
    ax.set_title('Pose-error distribution shifts across domains')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    save(fig, 'pose_error_distribution')


def plot_components(errs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, comp in zip(axes, ['E_T', 'E_R']):
        for d in DOMAINS:
            med = errs[d][comp].median()
            _log_hist(ax, errs[d][comp], DOMAIN_COLORS[d],
                      f'{d}  (median={med:.3g})')
            ax.axvline(med, color=DOMAIN_COLORS[d], ls='--', lw=1.4)
        thr = COMP_THRESHOLD[comp]
        # dotted (not dashed) so it reads clearly differently from the median lines
        ax.axvline(thr, color='k', ls=':', lw=1.8,
                   label=f'fail threshold ({thr:g} {COMP_UNIT[comp]})')
        ax.set_xlabel(f'{COMP_NAME[comp].capitalize()} Error  $E_{{{comp[-1]}}}$ '
                      f'[{COMP_UNIT[comp]}]  (Log Scale)')
        ax.set_ylabel('Density')
        ax.set_title(f'{COMP_NAME[comp].capitalize()} Error  ($E_{comp[-1]}$)')
        # explicit key for the dashed per-domain median lines, placed before the
        # dotted fail-threshold entry
        median_key = Line2D([0], [0], color='0.4', ls='--', lw=1.4,
                            label='median (per domain)')
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:-1] + [median_key] + handles[-1:],
                  frameon=False, fontsize=8.5)
        ax.grid(alpha=0.25, which='both')
    fig.suptitle('Domain Gap Splits into Translation and Rotation Error',
                 fontweight='bold')
    fig.tight_layout()
    save(fig, 'pose_error_components')


def main():
    errs = _load()
    error_stats(errs)
    plot_speed_score(errs)
    plot_components(errs)


if __name__ == '__main__':
    main()
