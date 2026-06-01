# src/threshold_stability.py
"""
How many unlabeled target-domain images do we need to set the failure threshold?

The deployable, label-free predictor flags an image as a likely SPNv2 failure when
its GMM anomaly score lands in the domain's top 30% (i.e. above the 70th percentile
of the score distribution). At test time we never see the full domain — we estimate
that 70th-percentile cut from a small unlabeled sample. This script bootstraps that
estimate to find the sample size n at which the cut stabilizes.

Quantities computed vs. n (the figure plots 1-2; 3 is saved to CSV for a later plot):
  1. Threshold value      — the 70th-pctile GMM score (mean ± std over resamples).
                            This is SHARED across error components: the cut lives on
                            the anomaly score, not on the error.
  2. Coverage             — fraction of the FULL domain that actually exceeds the
                            sample-estimated cut. Targets 0.30; scale-free because it
                            is rank-based, so invariant to any monotonic rescaling of
                            the score.
  3. Precision / recall   — of the predicted top-30% set against the TRUE top-30%
                            error set, computed PER COMPONENT (E_R, E_T). Labels are
                            used for evaluation only, never to fit the threshold.

Saves:
    figures/threshold_stability_gmm.png      -- panels 1-2
    results/NEW_DATA/threshold_stability_gmm.csv  -- all of 1-3
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'results' / 'NEW_DATA'
FIG_DIR      = PROJECT_ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

DETECTOR    = 'gmm'                 # use the GMM anomaly detector
DOMAINS     = ['lightbox', 'sunlamp']
COMPONENTS  = {'E_R': 'rotation [deg]', 'E_T': 'translation [m]'}

PCTILE      = 70.0                  # failure = top 30% → threshold at the 70th pctile
FAIL_FRAC   = 1.0 - PCTILE / 100.0  # 0.30 target coverage / failure base rate
N_GRID      = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_BOOT      = 2000
SEED        = 42

# stabilization criterion: smallest n where the realized coverage is both
#   accurate — |mean coverage - target 30%| < STABLE_TOL_BIAS  (no small-n over-flagging)
#   precise  — coverage std            < STABLE_TOL_COV         (low run-to-run wobble)
# Coverage is scale-free (rank-based), unlike the heavy-tailed raw threshold value, so
# both tolerances live in interpretable coverage-fraction units: "we flag 30% ± tol".
STABLE_TOL_COV  = 0.05   # precision: ±5 pts run-to-run scatter
STABLE_TOL_BIAS = 0.01   # accuracy:  mean within 1 pt of the 30% base rate

# domain colors matched to the plot_mahal_scores notebook
DOMAIN_COLORS = {'lightbox': '#4f0942', 'sunlamp': '#d6604d'}
COMP_STYLE    = {'E_R': '-', 'E_T': '--'}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_domain(domain: str):
    """Return (scores, {component: errors}) aligned by row."""
    scores = np.load(DATA_DIR / f'anomaly_scores_{DETECTOR}_{domain}.npy')
    df     = pd.read_csv(DATA_DIR / f'per_image_errors_{domain}.csv')
    n      = min(len(scores), len(df))
    scores = scores[:n]
    errors = {c: df[c].values[:n] for c in COMPONENTS}
    return scores, errors


# ── Bootstrap ───────────────────────────────────────────────────────────────────
def bootstrap_domain(scores: np.ndarray, errors: dict, rng: np.random.Generator):
    """
    For each n, draw N_BOOT samples of n scores (with replacement), set the cut at
    their 70th percentile, then evaluate that cut against the FULL domain.
    Returns a list of per-n metric dicts.
    """
    N = len(scores)
    # TRUE top-30% failure masks on the full domain, per component (eval only)
    true_fail = {c: e > np.percentile(e, PCTILE) for c, e in errors.items()}

    rows = []
    for n in N_GRID:
        if n > N:
            continue
        cuts      = np.empty(N_BOOT)
        coverage  = np.empty(N_BOOT)
        prec      = {c: np.empty(N_BOOT) for c in COMPONENTS}
        rec       = {c: np.empty(N_BOOT) for c in COMPONENTS}

        for b in range(N_BOOT):
            # Calibrate the failure threshold to this target domain: draw n unlabeled
            # images and set the cut at the 70th percentile of their scores. This is
            # the only test-time adaptation step — it tunes the cut to the domain's own
            # score distribution using no labels, just the n sampled scores.
            sample = scores[rng.integers(0, N, size=n)]
            cut    = np.percentile(sample, PCTILE)
            cuts[b] = cut

            pred = scores > cut                      # predicted-failure set on full domain
            coverage[b] = pred.mean()
            for c in COMPONENTS:
                tp = np.count_nonzero(pred & true_fail[c])
                prec[c][b] = tp / pred.sum() if pred.sum() else 0.0
                rec[c][b]  = tp / true_fail[c].sum() if true_fail[c].sum() else 0.0

        row = {
            'n': n,
            'cut_mean': cuts.mean(),   'cut_std': cuts.std(),
            'cov_mean': coverage.mean(), 'cov_std': coverage.std(),
        }
        for c in COMPONENTS:
            row[f'prec_{c}_mean'] = prec[c].mean(); row[f'prec_{c}_std'] = prec[c].std()
            row[f'rec_{c}_mean']  = rec[c].mean();  row[f'rec_{c}_std']  = rec[c].std()
        rows.append(row)

    return rows


def stabilization_n(df_dom: pd.DataFrame):
    """
    Smallest n where coverage is both accurate (|mean - target| < STABLE_TOL_BIAS)
    and precise (std < STABLE_TOL_COV). Returns int, or np.nan if no n qualifies.
    """
    ok = df_dom[(df_dom['cov_std'] < STABLE_TOL_COV) &
                ((df_dom['cov_mean'] - FAIL_FRAC).abs() < STABLE_TOL_BIAS)]
    return int(ok['n'].iloc[0]) if len(ok) else np.nan


# ── Plot ─────────────────────────────────────────────────────────────────────────
def plot(df: pd.DataFrame, stable: dict):
    """Clean two-panel threshold-stability figure: cut value + realized coverage vs n."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle('Number of Test Images Needed to Set Anomaly Score Failure Threshold\n'
                 'bands = ±1 SD across 2000 bootstrap resamples of n target images',
                 fontsize=12)

    # (left) threshold value vs n  — shared across components
    ax = axes[0]
    for dom in DOMAINS:
        d = df[df.domain == dom]
        ax.plot(d.n, d.cut_mean, marker='o', color=DOMAIN_COLORS[dom], label=dom)
        ax.fill_between(d.n, d.cut_mean - d.cut_std, d.cut_mean + d.cut_std,
                        color=DOMAIN_COLORS[dom], alpha=0.2)
        # label the selected threshold (most-converged cut, at the largest n)
        last = d.iloc[-1]
        ax.annotate(f'cut = {last.cut_mean:.2f}',
                    xy=(last.n, last.cut_mean), xytext=(8, 0),
                    textcoords='offset points', va='center', ha='left',
                    fontsize=9, color=DOMAIN_COLORS[dom], fontweight='bold')
        # stability line — the same n at which coverage stabilizes (where we read the cut)
        if not np.isnan(stable[dom]):
            ax.axvline(stable[dom], color=DOMAIN_COLORS[dom], ls=':', lw=1.2,
                       label=f'{dom} stable n≈{stable[dom]}')
    ax.set_xscale('log'); ax.set_xlabel('n unlabeled target images')
    ax.set_ylabel('70th-pctile GMM score (cut)')
    ax.set_title('Anomaly Score Threshold vs. Num Samples')
    ax.margins(x=0.18)   # room for the threshold labels at the right edge
    ax.legend(); ax.grid(alpha=0.3)

    # (right) realized coverage vs n  — should converge to the 30% base rate
    ax = axes[1]
    for dom in DOMAINS:
        d = df[df.domain == dom]
        ax.plot(d.n, d.cov_mean, marker='o', color=DOMAIN_COLORS[dom], label=dom)
        ax.fill_between(d.n, d.cov_mean - d.cov_std, d.cov_mean + d.cov_std,
                        color=DOMAIN_COLORS[dom], alpha=0.2)
        if not np.isnan(stable[dom]):
            ax.axvline(stable[dom], color=DOMAIN_COLORS[dom], ls=':', lw=1.2,
                       label=f'{dom} stable n≈{stable[dom]}')
    ax.axhline(FAIL_FRAC, color='k', ls='--', lw=1, label=f'target {FAIL_FRAC:.0%}')
    ax.set_xscale('log'); ax.set_xlabel('n unlabeled target images')
    ax.set_ylabel('fraction of domain flagged')
    ax.set_title(f'Realized Coverage Across Full Domain')
    # ax.set_title(f'Realized coverage (scale-free) — stable when '
    #              f'|bias| < {STABLE_TOL_BIAS:.0%} and SD < {STABLE_TOL_COV:.0%}')
    ax.legend(); ax.grid(alpha=0.3)

    # brief note: the cut is calibrated per domain from the sampled scores, label-free
    fig.text(0.5, -0.02,
             'Threshold = 70th percentile of the n sampled scores, calibrated per '
             'domain (no labels used).',
             ha='center', va='top', fontsize=9, style='italic', color='0.3')

    plt.tight_layout()
    out = FIG_DIR / 'threshold_stability_gmm.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → figures/{out.name}')


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    all_rows, stable = [], {}

    for dom in DOMAINS:
        scores, errors = load_domain(dom)
        rows = bootstrap_domain(scores, errors, rng)
        for r in rows:
            r['domain'] = dom
        df_dom = pd.DataFrame(rows)
        stable[dom] = stabilization_n(df_dom)
        all_rows.extend(rows)

        print(f"\n{'='*60}\n{dom}: {len(scores)} images")
        print(f"  coverage stabilizes (|bias| < {STABLE_TOL_BIAS:.0%} and std < "
              f"{STABLE_TOL_COV:.0%}) at n = {stable[dom]}")
        print(df_dom[['n', 'cut_mean', 'cut_std', 'cov_mean', 'cov_std']].to_string(index=False))

    df  = pd.DataFrame(all_rows)
    out = DATA_DIR / 'threshold_stability_gmm.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved → results/NEW_DATA/{out.name}")

    plot(df, stable)

    print(f"\n{'='*60}\nStabilization summary "
          f"(|bias| < {STABLE_TOL_BIAS:.0%} and coverage std < {STABLE_TOL_COV:.0%})")
    for dom in DOMAINS:
        print(f"  {dom:>9}: n ≈ {stable[dom]}")


if __name__ == '__main__':
    main()
