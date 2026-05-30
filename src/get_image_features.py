# src/get_image_features.py
"""
Compute interpretable image features / statistics for anomaly detection.
Features motivated by domain gap factors identified in:
- Park et al. 2022 (SPEED+): overexposure, texture, illumination
- Park et al. 2024 (SPNv2): brightness augmentation, style augmentation, sun flare

Saves:
    results/img_stats_{domain}.npy        -- (N, F) feature matrix
    results/filenames_stats_{domain}.npy  -- (N,)   filenames
    results/img_stats_feature_names.npy   -- (F,)   feature names
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import skew, entropy
from scipy.ndimage import laplace
from skimage.feature import local_binary_pattern
from multiprocessing import Pool, cpu_count

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT    = Path(os.environ.get('SPEEDPLUS_DATA', Path.home() / 'Desktop/speedplusv2'))
RESULTS_DIR  = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

DOMAINS = {
    'synthetic': ('labels/validation.csv', 'images'),
    'lightbox':  ('labels/test.csv',       'images'),
    'sunlamp':   ('labels/test.csv',       'images'),
}

FEATURE_NAMES = [
    # Illumination -- motivated by sunlamp overexposure (Park et al. 2022)
    'mean_brightness',
    'std_brightness',
    'overexposure_frac',    # fraction of pixels > 240
    'underexposure_frac',   # fraction of pixels < 15
    'brightness_skew',      # skewness of pixel intensity distribution
    'dynamic_range',        # max - min pixel value
    # Texture -- motivated by synthetic vs physical surface gap (SPNv2)
    'img_entropy',          # Shannon entropy of pixel histogram
    'lbp_variance',         # local binary pattern variance
    # Sharpness -- motivated by glare washing out edges (SPEED+)
    'sharpness_laplacian',  # Laplacian variance
    'freq_ratio',           # high / low frequency power ratio
    # Edge -- motivated by sun flare augmentation importance (SPNv2)
    'edge_density',         # mean gradient magnitude
    # Center region -- spacecraft is centered in SPEED+ by design
    'center_brightness',
    'center_contrast',
]

# ── Feature Computation ───────────────────────────────────────────────────────

def compute_stats(img_path: Path) -> np.ndarray:
    """
    Compute all image statistics for one image.
    Returns: (F,) float array
    """
    img = np.array(Image.open(img_path).convert('L')).astype(float)
    h, w = img.shape

    # ── Illumination ──────────────────────────────────────────────────────────
    mean_b         = img.mean()
    std_b          = img.std()
    overexposure   = (img > 240).mean()
    underexposure  = (img < 15).mean()
    bright_skew    = float(skew(img.flatten()))
    dynamic_range  = img.max() - img.min()

    # ── Texture ───────────────────────────────────────────────────────────────
    hist, _    = np.histogram(img, bins=256, range=(0, 256), density=True)
    img_ent    = float(entropy(hist + 1e-10))

    img_uint8  = img.astype(np.uint8)
    lbp        = local_binary_pattern(img_uint8, P=8, R=1, method='uniform')
    lbp_var    = float(lbp.var())

    # ── Sharpness ─────────────────────────────────────────────────────────────
    sharpness  = float(laplace(img).var())

    fft        = np.fft.fftshift(np.fft.fft2(img))
    power      = np.abs(fft) ** 2
    # after fftshift, low frequencies are in the center and high frequencies at the corners
    low_freq   = power[3*h//8 : 5*h//8, 3*w//8 : 5*w//8].mean()
    high_freq  = power[:h//4, :w//4].mean()
    freq_ratio = float(high_freq / (low_freq + 1.0))

    # ── Edge ─────────────────────────────────────────────────────────────────
    dx         = np.diff(img, axis=1)
    dy         = np.diff(img, axis=0)
    edge       = float(np.sqrt(dx[:dy.shape[0]]**2 +
                               dy[:, :dx.shape[1]]**2).mean())

    # ── Center crop ───────────────────────────────────────────────────────────
    cx, cy         = h // 2, w // 2
    crop           = img[cx - h//4 : cx + h//4, cy - w//4 : cy + w//4]
    center_bright  = float(crop.mean())
    center_contrast= float(crop.std())

    result = np.array([
        mean_b, std_b, overexposure, underexposure, bright_skew,
        dynamic_range, img_ent, lbp_var, sharpness, freq_ratio,
        edge, center_bright, center_contrast,
    ], dtype=np.float32)
    # bright_skew is nan for zero-variance images; replace any nan/inf with 0
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# ── Per-Domain Extraction ─────────────────────────────────────────────────────

def _compute_one(args):
    img_path, filename = args
    try:
        return filename, compute_stats(img_path)
    except Exception as e:
        print(f"Failed {filename}: {e}")
        return filename, None


def extract_domain(domain: str):
    csv_file, img_dir = DOMAINS[domain]
    csv_filenames = pd.read_csv(DATA_ROOT / domain / csv_file, header=None)[0].tolist()
    print(f"\n{domain}: {len(csv_filenames)} images")

    img_paths = [(DATA_ROOT / domain / img_dir / f, f) for f in csv_filenames]

    features, filenames = [], []
    with Pool(cpu_count()) as pool:
        for filename, feat in tqdm(pool.imap(_compute_one, img_paths), total=len(img_paths), desc=domain):
            if feat is not None:
                features.append(feat)
                filenames.append(filename)

    return np.array(features, dtype=np.float32), np.array(filenames)


def save_domain(features, filenames, domain: str):
    np.save(RESULTS_DIR / f'img_stats_{domain}.npy',       features)
    np.save(RESULTS_DIR / f'filenames_stats_{domain}.npy', filenames)
    print(f"Saved {features.shape} → results/img_stats_{domain}.npy")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # save feature names once for reference in analysis
    np.save(RESULTS_DIR / 'img_stats_feature_names.npy',
            np.array(FEATURE_NAMES))
    print(f"Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    for domain in DOMAINS:
        features, filenames = extract_domain(domain)
        save_domain(features, filenames, domain)

    print("\nDone.")


if __name__ == '__main__':
    main()