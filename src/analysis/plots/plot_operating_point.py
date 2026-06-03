"""
The threshold is chosen once on a synthetic validation split (F1-optimal), frozen,
and applied unchanged to each HIL domain. We report the precision and recall that
gate achieves per domain x axis: the numbers you'd get deploying it without ever
touching target labels. Uses model_outputs.synth_val_threshold.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.analysis._common import plt, RESULTS_DIR, COMP_NAME, save
from src.analysis.model_outputs import synth_val_threshold
from src.pipeline.models import load_domain, fail_labels, COMPONENTS, DOMAINS


def _table():
    rows, thresholds = [], {}
    for comp in COMPONENTS:
        t, clf = synth_val_threshold(comp)
        thresholds[comp] = t
        for dom in DOMAINS:
            X_hil, df_hil = load_domain(dom)
            yt = fail_labels(df_hil, comp)
            pred = (clf.predict_proba(X_hil)[:, 1] >= t).astype(int)
            rows.append({
                'domain': dom, 'component': comp, 'threshold': t,
                'precision': precision_score(yt, pred, zero_division=0),
                'recall':    recall_score(yt, pred, zero_division=0),
                'flagged_rate': pred.mean(),
                'fail_rate':    yt.mean(),
                'n': len(yt),
            })
    return pd.DataFrame(rows), thresholds


def plot(df):
    groups = [(d, c) for d in DOMAINS for c in COMPONENTS]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    prec = [df[(df.domain == d) & (df.component == c)].precision.iloc[0] for d, c in groups]
    rec = [df[(df.domain == d) & (df.component == c)].recall.iloc[0] for d, c in groups]
    b1 = ax.bar(x - w / 2, prec, w, color='#3182BD', label='precision')
    b2 = ax.bar(x + w / 2, rec,  w, color='#E6550D', label='recall')
    for bars, vals in ((b1, prec), (b2, rec)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=7.5)
    # chance precision per group = the domain's failure rate
    chance = [df[(df.domain == d) & (df.component == c)].fail_rate.iloc[0] for d, c in groups]
    ax.scatter(x - w / 2, chance, marker='_', s=420, color='k', zorder=5,
               label='chance precision (fail rate)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}\n{COMP_NAME[c]}' for d, c in groups])
    ax.set_ylabel('precision / recall')
    ax.set_ylim(0, 1.05)
    ax.set_title('Frozen synthetic-val threshold applied to HIL', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    save(fig, 'operating_point')


def main():
    df, thresholds = _table()
    out = RESULTS_DIR / 'operating_point.csv'
    df.to_csv(out, index=False)
    print('frozen thresholds (synthetic val, F1-optimal):',
          {c: round(t, 3) for c, t in thresholds.items()})
    print(df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'Saved -> results/{out.name}')
    plot(df)


if __name__ == '__main__':
    main()
