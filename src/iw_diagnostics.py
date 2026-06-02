# src/iw_diagnostics.py
"""
Stress-test the importance-weighting pipeline to decide whether IW *genuinely*
doesn't help, or whether our one configuration was just suppressing it.

Checks, per (domain, component):
  - supervised baseline AUC (no reweighting)
  - IW AUC across clip percentiles {90, 95, 99, 99.9, none}
  - IW AUC with self-normalized weights
  - IW AUC with the failure model UNweighted by class (isolate sample-weight effect)
  - effective sample size of the weights (variance cost of reweighting)
  - prediction agreement (corr) between supervised and IW probabilities
  - ORACLE ceiling: a classifier trained on HIL itself (cross-val) — tells us if
    there is any headroom at all, and whether covariate-shift (IW's assumption)
    can explain the gap.  Uses HIL labels ONLY as a diagnostic upper bound.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from importance_weighting import (
    load_domain, fail_labels, make_pipeline, _new_lr,
    domain_probabilities, COMPONENTS, DOMAINS, RANDOM_STATE,
)


def ess_fraction(w):
    """Kish effective sample size as a fraction of n: (Σw)² / (n·Σw²)."""
    w = np.asarray(w, float)
    return (w.sum() ** 2) / (len(w) * (w ** 2).sum())


def clipped(w, pct):
    return w if pct is None else np.minimum(w, np.percentile(w, pct))


def auc_iw(X_synth, ys, X_hil, yt, w, class_weight='balanced'):
    clf = make_pipeline(_new_lr(class_weight=class_weight))
    clf.fit(X_synth, ys, clf__sample_weight=w)
    return roc_auc_score(yt, clf.predict_proba(X_hil)[:, 1]), clf.predict_proba(X_hil)[:, 1]


def main():
    X_synth, df_synth = load_domain('synthetic')
    rows = []

    for domain in DOMAINS:
        X_hil, df_hil = load_domain(domain)
        p_oof, yd = domain_probabilities(X_synth, X_hil)
        p_synth = np.clip(p_oof[:len(X_synth)], 1e-6, 1 - 1e-6)
        w_raw = p_synth / (1 - p_synth)

        for comp in COMPONENTS:
            ys = fail_labels(df_synth, comp)
            yt = fail_labels(df_hil, comp)

            # baseline (no weights)
            sup = make_pipeline(_new_lr()).fit(X_synth, ys)
            p_sup = sup.predict_proba(X_hil)[:, 1]
            auc_sup = roc_auc_score(yt, p_sup)

            # oracle: train on HIL itself, cross-validated (upper bound / ceiling)
            cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
            p_oracle = cross_val_predict(make_pipeline(_new_lr()), X_hil, yt,
                                         cv=cv, method='predict_proba')[:, 1]
            auc_oracle = roc_auc_score(yt, p_oracle)

            rec = {'domain': domain, 'component': comp,
                   'supervised': auc_sup, 'oracle_HIL': auc_oracle}

            # clip sweep
            for pct in [90, 95, 99, 99.9, None]:
                w = clipped(w_raw, pct)
                auc, p_iw = auc_iw(X_synth, ys, X_hil, yt, w)
                rec[f'iw_clip{pct}'] = auc
                if pct == 99:
                    rec['ess_frac_clip99'] = ess_fraction(w)
                    rec['corr_sup_iw'] = np.corrcoef(p_sup, p_iw)[0, 1]

            # self-normalized weights (clip99 then scale to sum=n)
            w99 = clipped(w_raw, 99)
            w_sn = w99 * (len(w99) / w99.sum())
            rec['iw_selfnorm'], _ = auc_iw(X_synth, ys, X_hil, yt, w_sn)

            # failure model without class weighting (isolate the sample-weight effect)
            rec['iw_noclassw'], _ = auc_iw(X_synth, ys, X_hil, yt, w99, class_weight=None)

            rows.append(rec)

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200, 'display.max_columns', 30)

    print("\n=== AUC: supervised vs IW (clip sweep) vs oracle ceiling ===")
    cols = ['domain', 'component', 'supervised',
            'iw_clip90', 'iw_clip95', 'iw_clip99', 'iw_clip99.9', 'iw_clipNone',
            'iw_selfnorm', 'iw_noclassw', 'oracle_HIL']
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    print("\n=== headroom & weight diagnostics ===")
    df['iw_best'] = df[['iw_clip90', 'iw_clip95', 'iw_clip99', 'iw_clip99.9',
                        'iw_clipNone', 'iw_selfnorm', 'iw_noclassw']].max(axis=1)
    df['iw_best_minus_sup'] = df['iw_best'] - df['supervised']
    df['oracle_minus_sup'] = df['oracle_HIL'] - df['supervised']
    print(df[['domain', 'component', 'supervised', 'iw_best', 'iw_best_minus_sup',
              'oracle_HIL', 'oracle_minus_sup', 'ess_frac_clip99', 'corr_sup_iw']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
