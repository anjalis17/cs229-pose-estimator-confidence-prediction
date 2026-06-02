# src/analysis/run_all.py
"""
Regenerate every analysis figure for the writeup, in story order (parts 1–5).

    python src/analysis/run_all.py

Each plot can also be run on its own (e.g. python src/analysis/plot_roc_curves.py).
"""

# allow running directly: put repo root on sys.path so `import src` resolves
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.plots import (
    plot_error_distributions,   # part 1 — plots 1 & 2
    plot_feature_correlation,   # part 2 — plot 3
    plot_roc_curves,            # part 3 — plot 4
    plot_auc_bars,              # part 3 — plot 5
    plot_operating_point,       # part 4 — plot 6
    plot_risk_vs_error,         # part 4 — plot 7
    plot_iw_weights,            # part 5 — plot 8
)

STEPS = [
    ('part 1 — pose-error distribution + components', plot_error_distributions.main),
    ('part 2 — per-feature Spearman ρ',               plot_feature_correlation.main),
    ('part 3 — ROC curves (4 panels)',                plot_roc_curves.main),
    ('part 3 — AUC bar chart',                        plot_auc_bars.main),
    ('part 4 — operating point (frozen threshold)',   plot_operating_point.main),
    ('part 4 — predicted risk vs true error',         plot_risk_vs_error.main),
    ('part 5 — importance-weight histogram',          plot_iw_weights.main),
]


def main():
    for title, fn in STEPS:
        print(f'\n{"="*70}\n{title}\n{"="*70}')
        fn()
    print('\nAll analysis figures written to figures/.')


if __name__ == '__main__':
    main()
