"""
Run the trained models once per HIL domain and expose their per-image outputs.

Imports and reuses the fits from src/pipeline/models.py (no retraining logic
duplicated), so the analysis figures reflect exactly the deployed models. Per HIL
domain and error component it produces a per-image score for each of: method
(supervised LR), iw (importance-weighted LR), disagreement (the matching
multi-head feature), and oracle (cross-fit LR trained on HIL, a label upper
bound), plus the shared importance weights and domain-classifier AUC.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
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

DISAGREE_FEATURE = {'E_T': 'disagree_t_m', 'E_R': 'disagree_R_deg'}


def _oracle_probs(X_hil, yt):
    # out-of-fold HIL-trained probabilities; NaN if a component has only one class
    if len(np.unique(yt)) < 2:
        return np.full(len(yt), np.nan)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(make_pipeline(_new_lr()), X_hil, yt, cv=cv,
                             method='predict_proba')[:, 1]


def domain_outputs(domain, X_synth=None, df_synth=None):
    """Per-image arrays for one HIL domain: importance weights, domain AUC, reject
    mask, and per-component {y_true, error, probs:{method names}}."""
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
    X_synth, df_synth = load_domain('synthetic')
    return {d: domain_outputs(d, X_synth, df_synth) for d in DOMAINS}


def synth_val_threshold(comp, X_synth=None, df_synth=None, val_frac=0.25):
    """Honest operating point: fit supervised LR on a synthetic train split, freeze
    the F1-optimal threshold on the synthetic val split (HIL never seen). Returns
    (threshold, fitted_pipeline)."""
    if X_synth is None:
        X_synth, df_synth = load_domain('synthetic')
    ys = fail_labels(df_synth, comp)
    Xtr, Xval, ytr, yval = train_test_split(
        X_synth, ys, test_size=val_frac, stratify=ys, random_state=RANDOM_STATE)
    clf = make_pipeline(_new_lr()).fit(Xtr, ytr)

    p_val = clf.predict_proba(Xval)[:, 1]
    prec, rec, thr = precision_recall_curve(yval, p_val)
    # prec/rec have len(thr)+1; drop the last point to align f1 to thr
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    t = float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5
    return t, clf
