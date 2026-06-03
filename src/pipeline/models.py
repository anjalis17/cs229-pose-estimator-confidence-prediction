"""
Supervised + importance-weighted failure prediction under covariate shift.

Both classifiers are class-weighted logistic regressions trained ONLY on
synthetic failure labels; the importance-weighted version additionally reweights
synthetic samples toward the target HIL domain via density-ratio weights from a
domain classifier. HIL failure labels are used only for the final evaluation,
never in any fit. Evaluated against the baselines in baselines.py.

Per HIL domain (lightbox, sunlamp):
  1. fit a domain classifier (synthetic=0, HIL=1), cross-fit P(HIL|x)
  2. importance weight w = p/(1-p), clipped at the 99th percentile
  3. per component (E_R, E_T): weighted LR on synthetic, evaluate on HIL

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.loaders import load_features, KEEP_NAMES, RESULTS_DIR

PROJECT_ROOT = Path(__file__).parents[2]
FIG_DIR = PROJECT_ROOT / 'figures'

DOMAINS = ['lightbox', 'sunlamp']

# absolute failure thresholds [deg, m] (a fail is a fail, domain-independent)
COMPONENTS = {'E_R': 3.0, 'E_T': 0.10}
COMP_UNIT = {'E_R': 'deg', 'E_T': 'm'}

# heavy-tailed, strictly non-negative features: log1p before standardizing so a
# few extreme values don't dominate the domain classifier / importance weights
LOG_FEATURES = ['disagree_t_m', 'disagree_t_norm', 'seg_entropy', 'bbox_area']
LOG_IDX = [KEEP_NAMES.index(n) for n in LOG_FEATURES]

N_SPLITS = 5
WEIGHT_CLIP_PCT = 99
RANDOM_STATE = 42

DOMAIN_COLORS = {'lightbox': '#DD8452', 'sunlamp': '#55A868'}


def _log1p_transform(X):
    X = X.astype(float).copy()
    X[:, LOG_IDX] = np.log1p(np.clip(X[:, LOG_IDX], 0, None))
    return X


def make_pipeline(clf):
    # StandardScaler fits inside the pipeline, so each CV fold standardizes on its
    # own training data
    return Pipeline([
        ('log', FunctionTransformer(_log1p_transform)),
        ('scale', StandardScaler()),
        ('clf', clf),
    ])


def load_domain(domain):
    X = load_features(domain)
    df = pd.read_csv(RESULTS_DIR / f'per_image_errors_{domain}.csv')
    n = min(len(X), len(df))
    return X[:n], df.iloc[:n].reset_index(drop=True)


def fail_labels(df, comp):
    return (df[comp].values > COMPONENTS[comp]).astype(int)


def _new_lr(class_weight='balanced'):
    return LogisticRegression(max_iter=1000, class_weight=class_weight,
                              random_state=RANDOM_STATE)


def domain_probabilities(X_synth, X_hil):
    """Cross-fit P(HIL|x) over stacked synthetic+HIL. Returns (p_oof, y_domain)."""
    Xd = np.vstack([X_synth, X_hil])
    yd = np.r_[np.zeros(len(X_synth)), np.ones(len(X_hil))].astype(int)

    # unweighted LR here so P(HIL|x) stays calibrated; class_weight='balanced'
    # would inflate the probabilities and break the reliability curve. The weights
    # w = p/(1-p) still equal the density ratio up to a constant prior factor.
    pipe = make_pipeline(_new_lr(class_weight=None))
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    p_oof = cross_val_predict(pipe, Xd, yd, cv=cv, method='predict_proba')[:, 1]
    return p_oof, yd


def importance_weights(p_synth):
    eps = 1e-6
    p = np.clip(p_synth, eps, 1 - eps)
    w = p / (1 - p)
    cap = np.percentile(w, WEIGHT_CLIP_PCT)
    return np.minimum(w, cap), cap


def _metrics(domain, comp, method, y_true, y_prob):
    # threshold-free AUC + 0.5-cut F1/precision/recall (no HIL-label tuning)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'domain': domain,
        'component': comp,
        'method': method,
        'n_total': len(y_true),
        'n_fail': int(y_true.sum()),
        'fail_rate': y_true.mean(),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_component(domain, comp, X_synth, df_synth, X_hil, df_hil, w):
    ys = fail_labels(df_synth, comp)
    yt = fail_labels(df_hil, comp)
    rows = []

    # random: flag at the synthetic failure rate (deployable prior, HIL unseen).
    # Constant risk score -> AUC = 0.5.
    rng = np.random.default_rng(RANDOM_STATE)
    synth_rate = ys.mean()
    p_rand = np.full(len(yt), synth_rate)
    r = _metrics(domain, comp, 'random', yt, p_rand)
    r['auc'] = 0.5
    r['f1'] = f1_score(yt, (rng.random(len(yt)) < synth_rate).astype(int), zero_division=0)
    rows.append(r)

    # naive supervised: class-weighted LR on synthetic, applied to HIL (no adaptation)
    sup = make_pipeline(_new_lr()).fit(X_synth, ys)
    rows.append(_metrics(domain, comp, 'supervised', yt, sup.predict_proba(X_hil)[:, 1]))

    # ours: same fit but with sample_weight = w (the only difference is reweighting)
    iw = make_pipeline(_new_lr()).fit(X_synth, ys, clf__sample_weight=w)
    rows.append(_metrics(domain, comp, 'importance_weighted', yt, iw.predict_proba(X_hil)[:, 1]))

    return rows


def plot_calibration(calib):
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
    print('Saved -> figures/domain_calibration.(png|pdf)')


def plot_weights(weights):
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
    print('Saved -> figures/importance_weights.png')


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

        # stage 1: domain classifier (shared across components)
        p_oof, yd = domain_probabilities(X_synth, X_hil)
        auc_dom = roc_auc_score(yd, p_oof)
        p_synth = p_oof[:len(X_synth)]
        w, cap = importance_weights(p_synth)
        weights[domain] = w
        calib[domain] = calibration_curve(yd, p_oof, n_bins=10, strategy='quantile')
        print(f'  domain classifier AUC = {auc_dom:.3f}  (higher = bigger gap)')
        print(f'  importance weights: mean={w.mean():.2f}, max={w.max():.2f} '
              f'(clipped at p{WEIGHT_CLIP_PCT}={cap:.2f})')

        # stage 2: per component
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
    print(f'\nSaved -> results/importance_weighting_results.csv')

    plot_calibration(calib)
    plot_weights(weights)


if __name__ == '__main__':
    main()
