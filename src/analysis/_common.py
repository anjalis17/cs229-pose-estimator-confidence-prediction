# src/analysis/_common.py
"""Shared paths, color palette, and a figure-save helper for the analysis plots."""

import matplotlib
matplotlib.use('Agg')          # headless: scripts only write files, never show windows
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path         # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / 'results'
FIG_DIR      = PROJECT_ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# one color per domain (synthetic + the two HIL domains)
DOMAIN_COLORS = {'synthetic': '#4C72B0', 'lightbox': '#DD8452', 'sunlamp': '#55A868'}

# one color per method / score in the comparison plots
METHOD_COLORS = {
    'method':       '#2C7FB8',   # supervised LR  (our primary classifier)
    'iw':           '#7FCDBB',   # importance-weighted LR
    'disagreement': '#D95F0E',   # disagreement-threshold baseline
    'oracle':       '#444444',   # LR trained on HIL itself (label upper bound)
    'random':       '#BBBBBB',   # chance
}
METHOD_LABELS = {
    'method':       'supervised (ours)',
    'iw':           'importance-weighted (ours)',
    'disagreement': 'disagreement threshold',
    'oracle':       'oracle (HIL-trained)',
    'random':       'random',
}
# draw order for overlays (background → foreground)
METHOD_ORDER = ['random', 'disagreement', 'oracle', 'iw', 'method']

# error-component display
COMP_NAME = {'E_T': 'translation', 'E_R': 'rotation'}
COMP_UNIT = {'E_T': 'm', 'E_R': 'deg'}
COMP_THRESHOLD = {'E_T': 0.10, 'E_R': 3.0}   # absolute failure thresholds


def save(fig, name):
    """Save a figure to figures/ as both png and pdf."""
    for ext in ('png', 'pdf'):
        fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    print(f'Saved → figures/{name}.png|pdf')
