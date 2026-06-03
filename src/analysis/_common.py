"""
Shared paths, color palette, and a figure-save helper for the analysis plots.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import matplotlib
matplotlib.use('Agg')  # headless: scripts only write files, never show windows
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'
FIG_DIR = PROJECT_ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

DOMAIN_COLORS = {'synthetic': '#2166ac', 'lightbox': '#4f0942', 'sunlamp': '#d6604d'}

# one color per method / score in the comparison plots
METHOD_COLORS = {
    'method': '#2C7FB8',  # supervised LR (our primary classifier)
    'iw': '#7FCDBB',  # importance-weighted LR
    'disagreement': '#D95F0E',  # disagreement-threshold baseline
    'oracle': '#444444',  # LR trained on HIL itself (label upper bound)
    'random': '#BBBBBB',  # chance
}
METHOD_LABELS = {
    'method': 'supervised (ours)',
    'iw': 'importance-weighted (ours)',
    'disagreement': 'disagreement threshold',
    'oracle': 'oracle (HIL-trained)',
    'random': 'random',
}
# draw order for overlays (background to foreground)
METHOD_ORDER = ['random', 'disagreement', 'oracle', 'iw', 'method']

COMP_NAME = {'E_T': 'translation', 'E_R': 'rotation'}
COMP_UNIT = {'E_T': 'm', 'E_R': 'deg'}
COMP_THRESHOLD = {'E_T': 0.10, 'E_R': 3.0}  # absolute failure thresholds


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    print(f'Saved -> figures/{name}.png|pdf')
