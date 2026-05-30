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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)

# Config
PROJECT_ROOT  = Path(__file__).parent.parent
RESULTS_DIR   = PROJECT_ROOT / 'results'

# Threshold for binary fail/not-fail label
# Images with speed_score above this are considered failures
FAIL_THRESHOLD = 0.05

# Data Loading 
def load_features(domain: str, feature_type: str = 'img_stats') -> np.ndarray:
    """Load feature matrix for a domain. feature_type: 'img_stats' or 'gap'"""
    return np.load(RESULTS_DIR / f'{feature_type}_{domain}.npy')


def load_errors(domain: str) -> pd.DataFrame:
    """Load per-image SPNv2 errors from test.py output CSV."""
    return pd.read_csv(RESULTS_DIR / f'per_image_errors_{domain}.csv')

def report_split(y, name: str):
    """Print summary statistics about a binary label vector (e.g. a dataset split)"""
    y = np.asarray(y)
    n = len(y)
    fails = int(y.sum())
    succ = n - fails

    print(f"\n{name}")
    print(f"  Total:     {n}")
    print(f"  Successes: {succ} ({succ/n:.1%})")
    print(f"  Failures:  {fails} ({fails/n:.1%})")

def align_features_and_labels(features: np.ndarray,
                               errors: pd.DataFrame,
                               threshold: float = FAIL_THRESHOLD):
    """
    Pair feature vectors with binary fail/pass labels by index.

    features[i] and errors.iloc[i] correspond to the same image — both come
    from the _1000.csv evaluation set in the same order.
    """
    scores = errors['speed_score'].values
    n = min(len(features), len(scores))
    X = features[:n]
    y = (scores[:n] > threshold).astype(int) # 1 = fail, 0 = success
    return X, y


def print_metrics(y_true, y_pred, y_prob, name: str):
    n = len(y_true)
    correct = (y_true == y_pred).sum()
    print(f"\n── {name} ──")
    print(f"  Images:              {n}")
    print(f"  True failures:       {y_true.sum()} / {n}  ({y_true.mean():.1%})")
    print(f"  Predicted failures:  {y_pred.sum()} / {n}  ({y_pred.mean():.1%})")
    print(f"  Accuracy:            {correct} / {n}  ({correct/n:.1%})")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}  (of predicted failures, how many were real)")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}  (of real failures, how many we caught)")
    if y_prob is not None and len(np.unique(y_true)) > 1:
        print(f"  AUC-ROC:   {roc_auc_score(y_true, y_prob):.3f}  (0.5 = random, 1.0 = perfect)")


# ── Baseline 1: Random Classifier ─────────────────────────────────────────────
def random_baseline(y_true: np.ndarray, domain: str):
    """
    Predict failure with probability = empirical failure rate on test data.
    This is the performance floor (anything useful must beat this).
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
        'accuracy':       (y_true == y_pred).mean(),
        'f1':             f1_score(y_true, y_pred, zero_division=0),
        'precision':      precision_score(y_true, y_pred, zero_division=0),
        'recall':         recall_score(y_true, y_pred, zero_division=0),
        'auc':            roc_auc_score(y_true, y_prob)
                          if len(np.unique(y_true)) > 1 else np.nan,
    }


# ── Baseline 2: Supervised Classifier (Logistic Regression) ───────────────────

def supervised_baseline_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    domain:  str
):
    """
    Train logistic regression on synthetic train split, evaluate on synthetic
    val split (to show it fits) and on HIL test data (to show domain shift).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced') # balanced class weights to avoid overfitting to the majority class
    clf.fit(X_train_s, y_train)

    # Training accuracy (how well it fit)
    train_acc = (clf.predict(X_train_s) == y_train).mean()
    print(f"\nBaseline 2 (Supervised) — train accuracy on synthetic: {train_acc:.1%}")

    # Synthetic validation (in-distribution)
    y_val_pred = clf.predict(X_val_s)
    y_val_prob = clf.predict_proba(X_val_s)[:, 1]
    print_metrics(y_true=y_val, y_pred=y_val_pred, y_prob=y_val_prob,
                  name="Supervised [synthetic val]")

    # HIL test (out-of-distribution)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    print_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob,
                  name=f"Supervised [{domain}]")

    return [
        {
            'domain':    'synthetic val',
            'method':    'supervised',
            'accuracy':  (y_val == y_val_pred).mean(),
            'f1':        f1_score(y_val, y_val_pred, zero_division=0),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall':    recall_score(y_val, y_val_pred, zero_division=0),
            'auc':       roc_auc_score(y_val, y_val_prob)
                         if len(np.unique(y_val)) > 1 else np.nan,
        },
        {
            'domain':    domain,
            'method':    'supervised',
            'accuracy':  (y_test == y_pred).mean(),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'auc':       roc_auc_score(y_test, y_prob)
                         if len(np.unique(y_test)) > 1 else np.nan,
        },
    ]

# # ── Baseline 3: Supervised Classifier (Random Forest) ─────────────────────────
# def supervised_baseline_random_forest(
#     X_train: np.ndarray, y_train: np.ndarray,
#     X_val:   np.ndarray, y_val:   np.ndarray,
#     X_test:  np.ndarray, y_test:  np.ndarray,
#     domain:  str
# ):
#     """
#     Train random forest on synthetic train split, evaluate on synthetic val split and on HIL test data.

#     """
#     scaler = StandardScaler()
#     X_train_s = scaler.fit_transform(X_train)
#     X_val_s   = scaler.transform(X_val)
#     X_test_s  = scaler.transform(X_test)

#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train_s, y_train)

#     # Training accuracy (how well it fit)
#     train_acc = (clf.predict(X_train_s) == y_train).mean()
#     print(f"\nBaseline 3 (Supervised) — train accuracy on synthetic: {train_acc:.1%}")

#     # Synthetic validation (in-distribution)
#     y_val_pred = clf.predict(X_val_s)
#     y_val_prob = clf.predict_proba(X_val_s)[:, 1]
#     print_metrics(y_true=y_val, y_pred=y_val_pred, y_prob=y_val_prob,
#                   name="Supervised [synthetic val]")

#     # HIL test (out-of-distribution)
#     y_pred = clf.predict(X_test_s)
#     y_prob = clf.predict_proba(X_test_s)[:, 1]
#     print_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob,
#                   name=f"Supervised [{domain}]")

#     return [
#         {
#             'domain':    'synthetic val',
#             'method':    'supervised',
#             'accuracy':  (y_val == y_val_pred).mean(),
#             'f1':        f1_score(y_val, y_val_pred, zero_division=0),
#             'precision': precision_score(y_val, y_val_pred, zero_division=0),
#             'recall':    recall_score(y_val, y_val_pred, zero_division=0),
#             'auc':       roc_auc_score(y_val, y_val_prob)
#                          if len(np.unique(y_val)) > 1 else np.nan,
#         },
#         {
#             'domain':    domain,
#             'method':    'supervised',
#             'accuracy':  (y_test == y_pred).mean(),
#             'f1':        f1_score(y_test, y_pred, zero_division=0),
#             'precision': precision_score(y_test, y_pred, zero_division=0),
#             'recall':    recall_score(y_test, y_pred, zero_division=0),
#             'auc':       roc_auc_score(y_test, y_prob)
#                          if len(np.unique(y_test)) > 1 else np.nan,
#         },
#     ]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    feature_type = 'img_stats'  # switch to 'gap' (backbone activations of SPNv2) once we have that ready

    # Load synthetic data and split 800 train / 200 val
    X_synth   = load_features('synthetic', feature_type)
    err_synth = load_errors('synthetic')
    X_synth_al, y_synth = align_features_and_labels(X_synth, err_synth)

    X_train, y_train = X_synth_al[:9600], y_synth[:9600]
    X_val,   y_val   = X_synth_al[9600:], y_synth[9600:]
    report_split(y_train, "Synthetic Train")
    report_split(y_val, "Synthetic Val")

    print(f"Synthetic train: {len(X_train)} samples, failure rate: {y_train.mean():.3f}")
    print(f"Synthetic val:   {len(X_val)} samples,  failure rate: {y_val.mean():.3f}")

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Synthetic train split has only one class at FAIL_THRESHOLD={FAIL_THRESHOLD}. "
            "Lower the threshold."
        )

    all_results = []
    synth_val_added = False

    for domain in ['lightbox', 'sunlamp']:
        print(f"Domain: {domain}")

        # Load HIL data
        X_hil   = load_features(domain, feature_type)
        err_hil = load_errors(domain)
        X_hil_al, y_hil = align_features_and_labels(X_hil, err_hil)
        report_split(y_hil, f"{domain} Test")

        print(f"{domain}: {len(X_hil_al)} aligned samples")
        print(f"{domain} failure rate: {y_hil.mean():.3f}")

        if len(np.unique(y_hil)) < 2:
            print(f"Warning: only one class in {domain}, "
                  f"try adjusting FAIL_THRESHOLD ({FAIL_THRESHOLD})")
            continue

        # Baseline 1: Random
        r1 = random_baseline(y_hil, domain)
        all_results.append(r1)

        # Baseline 2: Supervised (trained on synthetic, tested on synthetic val + HIL)
        r2_list = supervised_baseline_logistic_regression(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_hil_al, y_test=y_hil,
            domain=domain
        )
        if not synth_val_added:
            all_results.append(r2_list[0])
            synth_val_added = True
        all_results.append(r2_list[1])

    # Summary table
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    df = pd.DataFrame(all_results)[
        ['domain', 'method', 'accuracy', 'f1', 'precision', 'recall', 'auc']
    ]
    print(df.to_string(index=False))

    # Save
    df.to_csv(RESULTS_DIR / 'baseline_results.csv', index=False)
    print(f"\nSaved → results/baseline_results.csv")


if __name__ == '__main__':
    main()