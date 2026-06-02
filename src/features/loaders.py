# src/features/loaders.py
"""
Shared feature loading for the failure-prediction pipeline.

Loads the SPNv2 model-internal features saved by
`src/features/get_model_features.py` and drops zero-variance / non-informative
columns before any model sees them.
"""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
RESULTS_DIR  = PROJECT_ROOT / 'results'

# Features dropped before fitting any model.
# 'reject' is constant (= 0) on all synthetic images → zero variance. It carries
# no in-distribution signal, so it is dropped from the standardized feature set.
DROP_FEATURES = ['reject']

# Saved feature names; we keep all but DROP_FEATURES.
ALL_FEATURE_NAMES = np.load(RESULTS_DIR / 'model_feature_names.npy', allow_pickle=True).tolist()
KEEP_IDX   = [i for i, n in enumerate(ALL_FEATURE_NAMES) if n not in DROP_FEATURES]
KEEP_NAMES = [ALL_FEATURE_NAMES[i] for i in KEEP_IDX]


def load_features(domain: str) -> np.ndarray:
    """Load the model features for a domain, dropping DROP_FEATURES."""
    X = np.load(RESULTS_DIR / f'model_features_{domain}.npy')
    return X[:, KEEP_IDX]
