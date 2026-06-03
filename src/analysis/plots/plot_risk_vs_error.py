"""
Scatter per domain x axis: x = the method's predicted failure probability,
y = the true pose error, colored by the frozen-threshold gate decision. If
predicted risk is meaningful, error rises with predicted probability and rejected
points concentrate above the physical failure line. Same supervised fit + frozen
synthetic-val threshold as the operating-point plot.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis._common import (
    plt, RESULTS_DIR, COMP_NAME, COMP_UNIT, COMP_THRESHOLD, save,
)
from src.analysis.model_outputs import synth_val_threshold
from src.pipeline.models import load_domain, COMPONENTS, DOMAINS

ACCEPT, REJECT = '#2CA02C', '#D62728'


def main():
    comps = ['E_T', 'E_R']
    # one frozen (threshold, classifier) per component, reused across domains
    gate = {c: synth_val_threshold(c) for c in comps}
    stats = []

    fig, axes = plt.subplots(len(DOMAINS), len(comps), figsize=(12, 10))
    for r, dom in enumerate(DOMAINS):
        X_hil, df_hil = load_domain(dom)
        for c, comp in enumerate(comps):
            ax = axes[r, c]
            t, clf = gate[comp]
            p = clf.predict_proba(X_hil)[:, 1]
            err = df_hil[comp].values
            reject = p >= t
            ythr = COMP_THRESHOLD[comp]

            stats.append({
                'domain': dom, 'component': comp, 'gate': t,
                'spearman_prob_vs_error': spearmanr(p, err).statistic,
                'n_reject': int(reject.sum()), 'n_accept': int((~reject).sum()),
                'mean_err_reject': float(err[reject].mean()) if reject.any() else np.nan,
                'mean_err_accept': float(err[~reject].mean()) if (~reject).any() else np.nan,
            })

            ax.scatter(p[~reject], err[~reject], s=8, c=ACCEPT, alpha=0.30,
                       linewidths=0, label='accept (pred pass)')
            ax.scatter(p[reject],  err[reject],  s=10, c=REJECT, alpha=0.45,
                       linewidths=0, label='reject (pred fail)')
            ax.axvline(t,    color='0.3', ls='--', lw=1.2)        # frozen gate
            ax.axhline(ythr, color='k',  ls=':',  lw=1.2)         # physical fail line
            ax.set_yscale('log')
            ax.set_xlim(0, 1)
            ax.set_title(f'{dom} - {COMP_NAME[comp]}  (gate={t:.2f})', fontsize=10)
            if r == len(DOMAINS) - 1:
                ax.set_xlabel('predicted failure probability')
            ax.set_ylabel(f'true {COMP_NAME[comp]} error [{COMP_UNIT[comp]}] (log)')
            ax.grid(alpha=0.2, which='both')
    handles = [plt.Line2D([], [], marker='o', ls='', color=ACCEPT, label='accept (pred pass)'),
               plt.Line2D([], [], marker='o', ls='', color=REJECT, label='reject (pred fail)'),
               plt.Line2D([], [], ls='--', color='0.3', label='frozen gate threshold'),
               plt.Line2D([], [], ls=':',  color='k',   label='physical failure threshold')]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('Predicted risk vs true error  (rejected points sit above the failure line)',
                 fontweight='bold', y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(RESULTS_DIR / 'risk_error_correlation.csv', index=False)
    print(df_stats.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('Saved -> results/risk_error_correlation.csv')
    save(fig, 'risk_vs_error')


if __name__ == '__main__':
    main()
