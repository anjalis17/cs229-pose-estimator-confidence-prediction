# Stanford CS 229 Final Project — Test-Time Confidence Prediction for Spacecraft Pose Estimation Under Domain Shift
Anjali Sreenivas (anjalisr), Lundeen Cahilly (lcahilly)

Predict, at test time, when SPNv2 spacecraft-pose predictions have failed, under the
synthetic → hardware-in-the-loop (HIL: `lightbox`, `sunlamp`) domain shift — using
features read out of the pose model itself. A failure is defined by absolute,
domain-independent thresholds:

- **translation:** `E_T > 0.10 m`
- **rotation:** `E_R > 3.0°`

## Setup

```bash
uv sync                      # create .venv from pyproject/uv.lock
source .venv/bin/activate    # or prefix commands with: uv run
```

## Repository layout

```
src/
  features/   get_model_features.py   # extract SPNv2 model-internal features from inference
              loaders.py              # shared feature loading (drops zero-variance 'reject')
              plot_feature_scales.py  # feature-preprocessing justification figure
  pipeline/   baselines.py            # dumb baselines: random + disagreement-threshold
              models.py               # our method: supervised + importance-weighted LR
              iw_diagnostics.py       # stress-test whether importance weighting helps
  utils/      errors.py               # pose-error / quaternion math
              data.py                 # SPEED+ label & camera loading
notebooks/    plot_estimator_error_dist.ipynb        # domain-gap pose-error figure
              plot_feature_error_correlation.ipynb   # feature ↔ error correlation figure
results/      *.npy / *.csv           # features, per-image errors, and outputs (gitignored)
figures/      *.png / *.pdf           # generated figures (gitignored)
```

Scripts run directly **from the repo root**:

```bash
python src/pipeline/models.py
```

## How to run everything

All commands are from the repo root. Steps 0–1 regenerate the feature matrix from
raw SPNv2 inference; if `results/model_features_*.npy` and
`results/per_image_errors_*.csv` already exist, **skip to step 2**.

| # | Command | Produces | Notes |
|---|---------|----------|-------|
| 0 | `bash scripts/run_test.sh` | `spnv2/.../predictions_pose.mat` per split | Runs SPNv2 inference. Needs the `spnv2/` submodule, the SPEED+ `data/`, and the model checkpoint. |
| 1 | `python src/features/get_model_features.py` | `results/model_features_{domain}.npy`, `results/model_feature_names.npy` | Reads the `predictions_pose.mat` from step 0. |
| 2 | `python src/features/plot_feature_scales.py` | `figures/feature_preprocessing.{png,pdf}` | Motivates dropping `reject` and standardizing. |
| 3 | `python src/pipeline/baselines.py` | `results/baseline_results.csv` | Random + disagreement baselines, AUC per domain × axis. |
| 4 | `python src/pipeline/models.py` | `results/importance_weighting_results.csv`, `figures/domain_calibration.{png,pdf}`, `figures/importance_weights.png` | Supervised + importance-weighted classifiers, evaluated on HIL. |
| 5 | `python src/pipeline/iw_diagnostics.py` | stdout only | Does IW beat plain supervised? Clip sweep + oracle ceiling. |

## Analysis figures (writeup, story order)

`src/analysis/` turns the model outputs into the figures for the writeup. Regenerate
all of them at once:

```bash
python src/analysis/run_all.py
```

or run any one on its own (e.g. `python src/analysis/plot_roc_curves.py`). Each
reuses the deployed fits from `models.py` / `baselines.py` via
`src/analysis/model_outputs.py` — no SPNv2 re-run, no retraining.

| Beat | Plot | Script | Figure |
|------|------|--------|--------|
| 1 — there's a domain gap | pose-error distribution | `plot_error_distributions.py` | `pose_error_distribution.*` |
| 1 | error split into translation/rotation | `plot_error_distributions.py` | `pose_error_components.*` |
| 2 — our features see the gap | per-feature Spearman ρ vs E_T/E_R | `plot_feature_correlation.py` | `feature_spearman.*` |
| 3 — method catches failures | ROC, 4 panels (method/IW/disagreement/oracle/random) | `plot_roc_curves.py` | `roc_curves.*` |
| 3 | AUC bars (method vs IW vs oracle) | `plot_auc_bars.py` | `auc_bars.*` (+ `results/analysis_auc.csv`) |
| 4 — usable gate, not just a ranker | operating point: synth-val threshold frozen → HIL P/R | `plot_operating_point.py` | `operating_point.*` (+ `results/operating_point.csv`) |
| 4 | predicted risk vs true error, accept/reject | `plot_risk_vs_error.py` | `risk_vs_error.*` |
| 5 — why IW is inert | importance-weight histogram p/(1−p) | `plot_iw_weights.py` | `iw_weights.*` |

Every script also writes its numbers to `results/` so the figures are backed by a
table you can quote directly:

| Stat CSV | Contents |
|----------|----------|
| `error_distribution_stats.csv` | per-domain mean/median/p90/p99 of E_T, E_R, SPEED score + failure rates |
| `feature_spearman.csv` | per-feature Spearman ρ vs E_T/E_R, per domain |
| `analysis_auc.csv` | AUC for method / IW / disagreement / oracle, per domain × axis |
| `operating_point.csv` | frozen threshold, precision, recall, flagged rate, fail rate per domain × axis |
| `risk_error_correlation.csv` | Spearman(pred prob, true error) + mean error of accepted vs rejected images |
| `iw_weight_stats.csv` | domain-classifier AUC (gap size), weight mean/max, effective sample size |

## What each result shows

- **`baseline_results.csv`** — per (domain, axis): `random` (AUC = 0.5 floor) and
  `disagreement` (the multi-head disagreement feature used directly as the failure
  score: `disagree_t_m` for translation, `disagree_R_deg` for rotation). Scored on
  `reject == 0` rows only (`disagree_*` is degenerate on PnP-rejected images).
- **`importance_weighting_results.csv`** — per (domain, component): `random`,
  `supervised` (class-weighted LR trained on synthetic), and `importance_weighted`
  (same, reweighted toward the target domain). AUC / F1 / precision / recall,
  evaluated on HIL labels (used for evaluation only — never for fitting).
- **`iw_diagnostics`** — whether importance weighting actually improves on the
  supervised classifier, vs. an oracle trained on HIL itself.
