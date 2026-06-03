"""
Compute model-internal features from SPNv2's forward pass for failure prediction.

These are read out of the pose model itself (pose / confidence space) rather than
the image domain, so they don't sign-flip across the synthetic->HIL gap the way
hand-engineered image stats did. Reads the predictions_pose.mat dumped by
inference (one per split) and saves a feature matrix + names per domain.

@ Author: Anjali Sreenivas and Lundeen Cahilly
@ Date: 2026-06-03
"""

import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.errors import quaternion_to_rotation_matrix, rotation_error_deg

PROJECT_ROOT = Path(__file__).parents[2]
RESULTS_DIR  = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

PRED_ROOT = PROJECT_ROOT / 'spnv2/tools/outputs/efficientdet_d3/full_config'

DOMAINS = ['synthetic', 'lightbox', 'sunlamp']

FEATURE_NAMES = [
    # multi-head pose disagreement (heatmap head vs EfficientPose head)
    'disagree_R_deg',       # geodesic angle between the two heads' rotations
    'disagree_t_m',         # ||heat_t - effi_t||
    'disagree_t_norm',      # same, normalized by mean target distance
    # model confidence read-outs
    'seg_entropy',          # mean foreground segmentation entropy
    'hm_entropy_mean',      # mean per-keypoint heatmap entropy
    'hm_peak_mean',         # mean peak activation height
    'hm_peak_min',          # worst-localized keypoint's peak height
    'hm_nconf',             # num keypoints above detection threshold
    'effi_cls',             # max EfficientPose detection confidence
    # free covariates
    'reject',               # heatmap PnP rejection flag
    'target_distance',      # ||effi_t||
    'bbox_area',
    'bbox_aspect',
]


def head_disagreement(heat_q, heat_t, effi_R, effi_t):
    N = heat_q.shape[0]
    dR      = np.full(N, np.nan)
    dt      = np.full(N, np.nan)
    dt_norm = np.full(N, np.nan)

    for i in range(N):
        q = heat_q[i]
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
            continue   # heatmap PnP rejected -> leave this row NaN
        R_heat = quaternion_to_rotation_matrix(q)
        dR[i] = rotation_error_deg(R_heat, effi_R[i])
        dt[i] = np.linalg.norm(heat_t[i] - effi_t[i])
        mean_dist  = np.linalg.norm(0.5 * (heat_t[i] + effi_t[i])) + 1e-9
        dt_norm[i] = dt[i] / mean_dist

    return dR, dt, dt_norm


def compute_features(m) -> np.ndarray:
    heat_q = np.asarray(m['heat_q'], dtype=float)   # (N, 4)
    heat_t = np.asarray(m['heat_t'], dtype=float)   # (N, 3)
    effi_R = np.asarray(m['effi_R'], dtype=float)   # (N, 3, 3)
    effi_t = np.asarray(m['effi_t'], dtype=float)   # (N, 3)
    bbox = np.asarray(m['bbox'], dtype=float)       # (N, 4) x1 y1 x2 y2

    dR, dt, dt_norm = head_disagreement(heat_q, heat_t, effi_R, effi_t)

    bw = np.abs(bbox[:, 2] - bbox[:, 0])
    bh = np.abs(bbox[:, 3] - bbox[:, 1])

    columns = {
        'disagree_R_deg': dR,
        'disagree_t_m': dt,
        'disagree_t_norm': dt_norm,
        'seg_entropy': np.asarray(m['seg_entropy'], dtype=float).ravel(),
        'hm_entropy_mean': np.asarray(m['hm_entropy_mean'], dtype=float).ravel(),
        'hm_peak_mean': np.asarray(m['hm_peak_mean'], dtype=float).ravel(),
        'hm_peak_min': np.asarray(m['hm_peak_min'], dtype=float).ravel(),
        'hm_nconf': np.asarray(m['hm_nconf'], dtype=float).ravel(),
        'effi_cls': np.asarray(m['effi_cls'], dtype=float).ravel(),
        'reject': np.asarray(m['reject'], dtype=float).ravel(),
        'target_distance': np.linalg.norm(effi_t, axis=1),
        'bbox_area': bw * bh,
        'bbox_aspect': bw / (bh + 1e-9),
    }

    features = np.column_stack([columns[name] for name in FEATURE_NAMES])
    return features.astype(np.float32)


def impute_nans(features, domain):
    # median-fill the disagreement columns on rejected rows; reject=1 stays set so
    # the failure signal is preserved
    for j, name in enumerate(FEATURE_NAMES):
        col = features[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            median = np.nanmedian(col)
            col[bad] = 0.0 if not np.isfinite(median) else median
            print(f"  {domain}: imputed {bad.sum()} non-finite in {name}")
    return features


def extract_domain(domain: str):
    mat_path = PRED_ROOT / domain / 'predictions_pose.mat'
    if not mat_path.exists():
        print(f"{domain}: missing {mat_path}, skipping")
        return None

    m = loadmat(mat_path, simplify_cells=True)
    features = compute_features(m)
    features = impute_nans(features, domain)
    print(f"{domain}: {features.shape[0]} images")
    return features


def save_domain(features, domain: str):
    np.save(RESULTS_DIR / f'model_features_{domain}.npy', features)
    print(f"Saved {features.shape} -> results/model_features_{domain}.npy")


def main():
    np.save(RESULTS_DIR / 'model_feature_names.npy', np.array(FEATURE_NAMES))
    print(f"Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    for domain in DOMAINS:
        features = extract_domain(domain)
        if features is not None:
            save_domain(features, domain)

    print("\nDone.")


if __name__ == '__main__':
    main()
