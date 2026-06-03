"""
Regenerate every analysis figure for the writeup, in story order.

    python src/analysis/run_all.py

Each plot can also be run on its own (e.g. python src/analysis/plots/plot_roc_curves.py).

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.plots import (
    plot_error_distributions,
    plot_feature_correlation,
    plot_roc_curves,
    plot_auc_bars,
    plot_operating_point,
    plot_risk_vs_error,
    plot_iw_weights,
)

STEPS = [
    ('part 1 - pose-error distribution + components', plot_error_distributions.main),
    ('part 2 - per-feature Spearman rho', plot_feature_correlation.main),
    ('part 3 - ROC curves (4 panels)', plot_roc_curves.main),
    ('part 3 - AUC bar chart', plot_auc_bars.main),
    ('part 4 - operating point (frozen threshold)', plot_operating_point.main),
    ('part 4 - predicted risk vs true error', plot_risk_vs_error.main),
    ('part 5 - importance-weight histogram', plot_iw_weights.main),
]


def main():
    for title, fn in STEPS:
        print(f'\n{"="*70}\n{title}\n{"="*70}')
        fn()
    print('\nAll analysis figures written to figures/.')


if __name__ == '__main__':
    main()
