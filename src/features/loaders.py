"""
Shared feature loading for the failure-prediction pipeline.

Loads the SPNv2 model-internal features saved by get_model_features.py and drops
zero-variance / non-informative columns before our models see them.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'

# 'reject' is constant (=0) on synthetic, so it has zero variance / no signal there
DROP_FEATURES = ['reject']

ALL_FEATURE_NAMES = np.load(RESULTS_DIR / 'model_feature_names.npy', allow_pickle=True).tolist()
KEEP_IDX = [i for i, n in enumerate(ALL_FEATURE_NAMES) if n not in DROP_FEATURES]
KEEP_NAMES = [ALL_FEATURE_NAMES[i] for i in KEEP_IDX]


def load_features(domain):
    X = np.load(RESULTS_DIR / f'model_features_{domain}.npy')
    return X[:, KEEP_IDX]
