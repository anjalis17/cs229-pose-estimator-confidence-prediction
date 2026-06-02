# src/pipeline/models.py
"""
Our method: supervised + importance-weighted failure prediction (covariate-shift
adaptation).

Both classifiers are class-weighted logistic regressions trained ONLY on
synthetic failure labels; the importance-weighted version additionally reweights
synthetic samples toward the target HIL domain. They are evaluated against the
dumb baselines in src/pipeline/baselines.py.

The failure classifier only ever trains on SYNTHETIC failure labels. HIL failure
labels are used exclusively for the final evaluation -- never in any fit. To make
the synthetic-trained model transfer to a HIL domain, we reweight synthetic
samples by how much they "look like" that HIL domain (density-ratio importance
weights), estimated by a domain classifier.

Per test domain ∈ {lightbox, sunlamp}:
  1. Domain dataset: synthetic → 0, this HIL domain → 1.
  2. Class-weighted logistic regression predicting synthetic-vs-HIL.
  3. Cross-fit (out-of-fold) probabilities p = P(HIL | x) so every synthetic
     image gets an *unbiased* p from a model that never trained on it.
     → reliability/calibration curve (p vs. empirical domain fraction)
     → importance weight  w = p / (1 - p),  clipped at the 99th percentile.
  4. Per component ∈ {E_R, E_T}:
       - synthetic failure labels from an absolute threshold,
       - weighted logistic regression on synthetic (sample_weight = w),
       - evaluate on HIL, alongside random + naive-supervised baselines.

Outputs:
  results/importance_weighting_results.csv
  figures/domain_calibration.png
  figures/importance_weights.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)

# allow running this file directly (python src/pipeline/models.py) by putting
# the repo root on sys.path so `import src` resolves
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.loaders import load_features, KEEP_NAMES, RESULTS_DIR

PROJECT_ROOT = Path(__file__).parents[2]
FIG_DIR      = PROJECT_ROOT / 'figures'

DOMAINS = ['lightbox', 'sunlamp']

# Absolute failure thresholds (a fail is a fail, domain-independent).
COMPONENTS = {'E_R': 3.0, 'E_T': 0.10}   # [deg, m]
COMP_UNIT  = {'E_R': 'deg', 'E_T': 'm'}

# Heavy-tailed, strictly non-negative features → log1p before standardizing so a
# few extreme values don't dominate the domain classifier / importance weights.
LOG_FEATURES = ['disagree_t_m', 'disagree_t_norm', 'seg_entropy', 'bbox_area']
LOG_IDX = [KEEP_NAMES.index(n) for n in LOG_FEATURES]

N_SPLITS        = 5
WEIGHT_CLIP_PCT = 99
RANDOM_STATE    = 42

DOMAIN_COLORS = {'lightbox': '#DD8452', 'sunlamp': '#55A868'}


# ----------------------------------------------------------------------------
# Preprocessing + data loading
# ----------------------------------------------------------------------------
def _log1p_transform(X):
    """log1p the heavy-tailed columns; leave the rest untouched."""
    X = X.astype(float).copy()
    X[:, LOG_IDX] = np.log1p(np.clip(X[:, LOG_IDX], 0, None))
    return X


def make_pipeline(clf):
    """log1p → standardize → classifier. StandardScaler fits inside the pipeline,
    so under cross-validation each fold standardizes on its own training data."""
    return Pipeline([
        ('log',   FunctionTransformer(_log1p_transform)),
        ('scale', StandardScaler()),
        ('clf',   clf),
    ])


def load_domain(domain):
    """Return (X_12feat, errors_df) aligned to the same length."""
    X  = load_features(domain)                                   # (n1, 12)
    df = pd.read_csv(RESULTS_DIR / f'per_image_errors_{domain}.csv')
    n  = min(len(X), len(df))
    return X[:n], df.iloc[:n].reset_index(drop=True)


def fail_labels(df, comp):
    """Binary failure labels from the absolute per-component threshold."""
    return (df[comp].values > COMPONENTS[comp]).astype(int)


def _new_lr(class_weight='balanced'):
    return LogisticRegression(max_iter=1000, class_weight=class_weight,
                              random_state=RANDOM_STATE)


# ----------------------------------------------------------------------------
# Stage 1: domain classifier → out-of-fold p → importance weights
# ----------------------------------------------------------------------------
def domain_probabilities(X_synth, X_hil):
    """Cross-fit P(HIL | x) for the stacked synthetic+HIL set.

    Returns (p_oof, y_domain): out-of-fold HIL-probabilities and the true domain
    labels (0 = synthetic, 1 = HIL), for every row of np.vstack([X_synth, X_hil]).
    """
    Xd = np.vstack([X_synth, X_hil])
    yd = np.r_[np.zeros(len(X_synth)), np.ones(len(X_hil))].astype(int)

    # Unweighted LR for the domain classifier: we want CALIBRATED P(HIL | x) so
    # the reliability curve is meaningful. class_weight='balanced' would inflate
    # predicted probabilities and break calibration. The resulting weights
    # w = p/(1-p) still equal the density ratio up to a constant prior factor,
    # which scales all weights uniformly and does not affect the failure model.
    pipe = make_pipeline(_new_lr(class_weight=None))
    cv   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    p_oof = cross_val_predict(pipe, Xd, yd, cv=cv, method='predict_proba')[:, 1]
    return p_oof, yd


def importance_weights(p_synth):
    """w = p/(1-p), clipped at the WEIGHT_CLIP_PCT percentile for stability."""
    eps = 1e-6
    p   = np.clip(p_synth, eps, 1 - eps)
    w   = p / (1 - p)
    cap = np.percentile(w, WEIGHT_CLIP_PCT)
    return np.minimum(w, cap), cap


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _metrics(domain, comp, method, y_true, y_prob):
    """Threshold-free AUC + 0.5-cut F1/precision/recall (no HIL-label tuning)."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'domain':    domain,
        'component': comp,
        'method':    method,
        'n_total':   len(y_true),
        'n_fail':    int(y_true.sum()),
        'fail_rate': y_true.mean(),
        'auc':       roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
    }


# ----------------------------------------------------------------------------
# Stage 2: per-component failure classifiers (+ baselines)
# ----------------------------------------------------------------------------
def evaluate_component(domain, comp, X_synth, df_synth, X_hil, df_hil, w):
    """Random + naive-supervised + importance-weighted, evaluated on HIL."""
    ys = fail_labels(df_synth, comp)
    yt = fail_labels(df_hil,   comp)
    rows = []

    # Baseline 1 — random: predict at the empirical HIL failure rate (AUC = 0.5).
    rng    = np.random.default_rng(RANDOM_STATE)
    p_rand = np.full(len(yt), yt.mean())
    r = _metrics(domain, comp, 'random', yt, p_rand)
    r['auc'] = 0.5
    r['f1']  = f1_score(yt, (rng.random(len(yt)) < yt.mean()).astype(int), zero_division=0)
    rows.append(r)

    # Baseline 2 — naive supervised: class-weighted LR on synthetic, applied to HIL
    #               (no domain adaptation).
    sup = make_pipeline(_new_lr()).fit(X_synth, ys)
    rows.append(_metrics(domain, comp, 'supervised', yt, sup.predict_proba(X_hil)[:, 1]))

    # Ours — importance-weighted: identical to baseline 2 but with sample_weight = w,
    #         so the only difference is the covariate-shift reweighting.
    iw = make_pipeline(_new_lr()).fit(X_synth, ys, clf__sample_weight=w)
    rows.append(_metrics(domain, comp, 'importance_weighted', yt, iw.predict_proba(X_hil)[:, 1]))

    return rows


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_calibration(calib):
    """Reliability diagram of the domain classifier, one line per HIL domain."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], '--', color='0.6', label='perfectly calibrated')
    for domain, (frac_pos, mean_pred) in calib.items():
        ax.plot(mean_pred, frac_pos, 'o-', color=DOMAIN_COLORS[domain],
                label=f'{domain} domain classifier')
    ax.set_xlabel('Mean predicted P(HIL | x)  [out-of-fold]')
    ax.set_ylabel('Empirical fraction that is HIL')
    ax.set_title('Domain-classifier calibration')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal')
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / 'domain_calibration.png', dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'domain_calibration.pdf', bbox_inches='tight')
    print('Saved → figures/domain_calibration.(png|pdf)')


def plot_weights(weights):
    """Distribution of synthetic importance weights, one panel per HIL domain."""
    fig, axes = plt.subplots(1, len(weights), figsize=(4.4 * len(weights), 3.4),
                             squeeze=False)
    for ax, (domain, w) in zip(axes[0], weights.items()):
        ax.hist(w, bins=60, color=DOMAIN_COLORS[domain], alpha=0.8, edgecolor='none')
        ax.axvline(1.0, color='0.4', ls='--', lw=1, label='w = 1 (no reweighting)')
        ax.set_yscale('log')
        ax.set_xlabel('importance weight  w = p/(1-p)')
        ax.set_ylabel('count (log)')
        ax.set_title(f'{domain}  (mean={w.mean():.2f}, max={w.max():.2f})')
        ax.legend(fontsize=8)
    fig.suptitle('Synthetic importance weights toward each HIL domain', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'importance_weights.png', dpi=200, bbox_inches='tight')
    print('Saved → figures/importance_weights.png')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    X_synth, df_synth = load_domain('synthetic')
    print(f'Synthetic: {len(X_synth)} images, {X_synth.shape[1]} features')
    for comp, thr in COMPONENTS.items():
        ys = fail_labels(df_synth, comp)
        print(f'  synthetic {comp} > {thr}{COMP_UNIT[comp]}: '
              f'fail rate {ys.mean():.1%} ({ys.sum()} fails)')

    all_rows, calib, weights = [], {}, {}

    for domain in DOMAINS:
        print(f'\n{"="*64}\nDomain: {domain}\n{"="*64}')
        X_hil, df_hil = load_domain(domain)

        # Stage 1 — domain classifier (shared across components)
        p_oof, yd = domain_probabilities(X_synth, X_hil)
        auc_dom   = roc_auc_score(yd, p_oof)
        p_synth   = p_oof[:len(X_synth)]
        w, cap    = importance_weights(p_synth)
        weights[domain] = w
        calib[domain]   = calibration_curve(yd, p_oof, n_bins=10, strategy='quantile')
        print(f'  domain classifier AUC = {auc_dom:.3f}  (higher = bigger gap)')
        print(f'  importance weights: mean={w.mean():.2f}, max={w.max():.2f} '
              f'(clipped at p{WEIGHT_CLIP_PCT}={cap:.2f})')

        # Stage 2 — per component
        for comp in COMPONENTS:
            all_rows += evaluate_component(domain, comp, X_synth, df_synth,
                                           X_hil, df_hil, w)

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / 'importance_weighting_results.csv'
    df.to_csv(out_csv, index=False)

    print(f'\n{"="*64}\nSUMMARY (evaluated on HIL)\n{"="*64}')
    show = df[['domain', 'component', 'method', 'auc', 'f1', 'precision',
               'recall', 'fail_rate']].copy()
    print(show.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nSaved → results/importance_weighting_results.csv')

    plot_calibration(calib)
    plot_weights(weights)


if __name__ == '__main__':
    main()
