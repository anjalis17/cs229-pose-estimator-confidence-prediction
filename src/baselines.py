# src/baselines.py
"""
Baselines for confidence prediction:
1. Random classifier -- predicts failure at empirical HIL failure rate
2. Supervised classifier -- trained on synthetic features + SPNv2 error labels,
   applied directly to HIL features (expected to fail due to domain shift)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
RESULTS_DIR   = PROJECT_ROOT / 'results'

# Threshold for binary fail/not-fail label
# Images with speed_score above this are considered failures
FAIL_THRESHOLD = 0.5

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_features(domain: str, feature_type: str = 'img_stats') -> np.ndarray:
    """Load feature matrix for a domain. feature_type: 'img_stats' or 'gap'"""
    return np.load(RESULTS_DIR / f'{feature_type}_{domain}.npy')


def load_filenames(domain: str, feature_type: str = 'img_stats') -> np.ndarray:
    prefix = 'filenames_stats' if feature_type == 'img_stats' else 'filenames'
    return np.load(RESULTS_DIR / f'{prefix}_{domain}.npy')


def load_errors(domain: str) -> pd.DataFrame:
    """Load per-image SPNv2 errors from test.py output CSV."""
    return pd.read_csv(RESULTS_DIR / f'per_image_errors_{domain}.csv')


def make_binary_labels(errors: pd.DataFrame,
                       filenames: np.ndarray,
                       threshold: float = FAIL_THRESHOLD):
    """
    Join error data to feature filenames, return binary fail labels.
    1 = failure (speed_score > threshold), 0 = success
    """
    error_map = dict(zip(errors['filename'], errors['speed_score']))
    labels = np.array([
        1 if error_map.get(f, np.nan) > threshold else 0
        for f in filenames
    ])
    # mask out any filenames not found in error data
    valid = np.array([f in error_map for f in filenames])
    return labels[valid], valid


def align_features_and_labels(features, filenames, errors, threshold=FAIL_THRESHOLD):
    """
    Align feature matrix with error labels by filename.
    Returns aligned (X, y) arrays.
    """
    error_map = dict(zip(errors['filename'], errors['speed_score']))

    X, y = [], []
    for i, fname in enumerate(filenames):
        if fname in error_map:
            X.append(features[i])
            score = error_map[fname]
            y.append(1 if score > threshold else 0)

    return np.array(X), np.array(y)


def print_metrics(y_true, y_pred, y_prob, name: str):
    print(f"\n── {name} ──")
    print(f"  Failure rate in data: {y_true.mean():.3f}")
    print(f"  Predicted failure rate: {y_pred.mean():.3f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    if y_prob is not None and len(np.unique(y_true)) > 1:
        print(f"  AUC-ROC:   {roc_auc_score(y_true, y_prob):.3f}")
    print(classification_report(y_true, y_pred,
                                 target_names=['success', 'failure'],
                                 zero_division=0))


# ── Baseline 1: Random Classifier ─────────────────────────────────────────────

def random_baseline(y_true: np.ndarray, domain: str):
    """
    Predict failure with probability = empirical failure rate in HIL data.
    This is the performance floor -- anything useful must beat this.
    """
    failure_rate = y_true.mean()
    np.random.seed(42)
    y_pred = (np.random.rand(len(y_true)) < failure_rate).astype(int)
    y_prob = np.full(len(y_true), failure_rate)

    print(f"\nBaseline 1 (Random) on {domain}:")
    print(f"  Empirical failure rate: {failure_rate:.3f}")
    print_metrics(y_true, y_pred, y_prob, f"Random [{domain}]")

    return {
        'domain':         domain,
        'method':         'random',
        'failure_rate':   failure_rate,
        'f1':             f1_score(y_true, y_pred, zero_division=0),
        'precision':      precision_score(y_true, y_pred, zero_division=0),
        'recall':         recall_score(y_true, y_pred, zero_division=0),
        'auc':            roc_auc_score(y_true, y_prob)
                          if len(np.unique(y_true)) > 1 else np.nan,
    }


# ── Baseline 2: Supervised Classifier ─────────────────────────────────────────

def supervised_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    domain:  str
):
    """
    Train logistic regression on synthetic features + SPNv2 error labels.
    Apply directly to HIL features.
    Expected to fail due to domain shift in feature space.
    """
    # Normalize features
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    print(f"\nBaseline 2 (Supervised) on {domain}:")
    print_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob,
                  name=f"Supervised [{domain}]")

    return {
        'domain':    domain,
        'method':    'supervised',
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'auc':       roc_auc_score(y_test, y_prob)
                     if len(np.unique(y_test)) > 1 else np.nan,
        'clf':       clf,
        'scaler':    scaler,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    feature_type = 'img_stats'  # switch to 'gap' when backbone features ready

    # Load synthetic data for training supervised baseline
    X_synth     = load_features('synthetic', feature_type)
    f_synth     = load_filenames('synthetic', feature_type)
    err_synth   = load_errors('synthetic')
    X_synth_al, y_synth = align_features_and_labels(X_synth, f_synth, err_synth)

    print(f"Synthetic: {len(X_synth_al)} aligned samples")
    print(f"Synthetic failure rate: {y_synth.mean():.3f}")

    all_results = []

    for domain in ['lightbox', 'sunlamp']:
        print(f"\n{'='*50}")
        print(f"Domain: {domain}")
        print(f"{'='*50}")

        # Load HIL data
        X_hil   = load_features(domain, feature_type)
        f_hil   = load_filenames(domain, feature_type)
        err_hil = load_errors(domain)
        X_hil_al, y_hil = align_features_and_labels(X_hil, f_hil, err_hil)

        print(f"{domain}: {len(X_hil_al)} aligned samples")
        print(f"{domain} failure rate: {y_hil.mean():.3f}")

        if len(np.unique(y_hil)) < 2:
            print(f"Warning: only one class in {domain}, "
                  f"try adjusting FAIL_THRESHOLD ({FAIL_THRESHOLD})")
            continue

        # Baseline 1: Random
        r1 = random_baseline(y_hil, domain)
        all_results.append(r1)

        # Baseline 2: Supervised (trained on synthetic, tested on HIL)
        r2 = supervised_baseline(
            X_train=X_synth_al, y_train=y_synth,
            X_test=X_hil_al,    y_test=y_hil,
            domain=domain
        )
        all_results.append(r2)

    # Summary table
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    df = pd.DataFrame(all_results)[
        ['domain', 'method', 'f1', 'precision', 'recall', 'auc']
    ]
    print(df.to_string(index=False))

    # Save
    df.to_csv(RESULTS_DIR / 'baseline_results.csv', index=False)
    print(f"\nSaved → results/baseline_results.csv")


if __name__ == '__main__':
    main()