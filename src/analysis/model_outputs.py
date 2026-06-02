# src/analysis/model_outputs.py
"""
Run the two trained models once per HIL domain and expose their PER-IMAGE outputs.

These are the same fits as src/pipeline/models.py — we import its functions and
reuse them, so the analysis figures reflect exactly the deployed models (no
retraining logic is duplicated here). For each HIL domain and error component we
produce a predicted score per image:

  method        supervised class-weighted LR (synthetic → HIL)
  iw            importance-weighted LR (covariate-shift reweighted)
  disagreement  the matching multi-head disagreement feature, used directly as
                the score (disagree_t_m for E_T, disagree_R_deg for E_R)
  oracle        cross-fit LR trained on HIL itself — a label upper bound

plus the shared synthetic importance weights and the domain-classifier AUC.

`synth_val_threshold` additionally gives the honest operating point: train on a
synthetic train split, freeze the F1-optimal threshold on the synthetic val
split, never looking at HIL.
"""

import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve

from src.pipeline.models import (
    load_domain, fail_labels, make_pipeline, _new_lr,
    domain_probabilities, importance_weights,
    COMPONENTS, DOMAINS, N_SPLITS, RANDOM_STATE,
)
from src.features.loaders import KEEP_NAMES

# score each component with its own multi-head disagreement feature
DISAGREE_FEATURE = {'E_T': 'disagree_t_m', 'E_R': 'disagree_R_deg'}


def _oracle_probs(X_hil, yt):
    """Out-of-fold HIL-trained probabilities — each row scored by a model that
    never trained on it. NaN if a component has only one class present."""
    if len(np.unique(yt)) < 2:
        return np.full(len(yt), np.nan)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(make_pipeline(_new_lr()), X_hil, yt, cv=cv,
                             method='predict_proba')[:, 1]


def domain_outputs(domain, X_synth=None, df_synth=None):
    """Per-image arrays for one HIL domain.

    Returns dict:
      'weights'    (n_synth,)  importance weights p/(1-p)
      'domain_auc' float       synthetic-vs-HIL classifier AUC (gap size)
      'reject'     (n_hil,)    bool, PnP-rejected rows
      comp: {'y_true', 'error', 'probs': {name: (n_hil,) score}}  for comp in COMPONENTS
    """
    if X_synth is None:
        X_synth, df_synth = load_domain('synthetic')
    X_hil, df_hil = load_domain(domain)

    p_oof, yd = domain_probabilities(X_synth, X_hil)
    domain_auc = roc_auc_score(yd, p_oof)
    w, _ = importance_weights(p_oof[:len(X_synth)])

    out = {'weights': w, 'domain_auc': domain_auc,
           'reject': df_hil['rejected'].values.astype(bool)}

    feat_idx = {c: KEEP_NAMES.index(DISAGREE_FEATURE[c]) for c in COMPONENTS}
    for comp in COMPONENTS:
        ys = fail_labels(df_synth, comp)
        yt = fail_labels(df_hil, comp)
        probs = {
            'method':       make_pipeline(_new_lr()).fit(X_synth, ys)
                                .predict_proba(X_hil)[:, 1],
            'iw':           make_pipeline(_new_lr()).fit(X_synth, ys, clf__sample_weight=w)
                                .predict_proba(X_hil)[:, 1],
            'disagreement': X_hil[:, feat_idx[comp]],
            'oracle':       _oracle_probs(X_hil, yt),
        }
        out[comp] = {'y_true': yt, 'error': df_hil[comp].values, 'probs': probs}
    return out


def all_domain_outputs():
    """domain_outputs for every HIL domain, sharing one synthetic load/fit."""
    X_synth, df_synth = load_domain('synthetic')
    return {d: domain_outputs(d, X_synth, df_synth) for d in DOMAINS}


def synth_val_threshold(comp, X_synth=None, df_synth=None, val_frac=0.25):
    """Honest operating point: fit supervised LR on a synthetic TRAIN split, then
    freeze the F1-optimal probability threshold on the synthetic VAL split.
    HIL is never seen. Returns (threshold, fitted_pipeline_trained_on_train_split)."""
    if X_synth is None:
        X_synth, df_synth = load_domain('synthetic')
    ys = fail_labels(df_synth, comp)
    Xtr, Xval, ytr, yval = train_test_split(
        X_synth, ys, test_size=val_frac, stratify=ys, random_state=RANDOM_STATE)
    clf = make_pipeline(_new_lr()).fit(Xtr, ytr)

    p_val = clf.predict_proba(Xval)[:, 1]
    prec, rec, thr = precision_recall_curve(yval, p_val)
    # prec/rec have length len(thr)+1; align f1 to thr by dropping the last point
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    t = float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5
    return t, clf
