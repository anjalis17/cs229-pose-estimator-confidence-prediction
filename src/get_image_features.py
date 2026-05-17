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

import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import skew, entropy
from scipy.ndimage import laplace
from skimage.feature import local_binary_pattern

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT    = PROJECT_ROOT / 'data/speedplusv2/speedplus'
RESULTS_DIR  = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLES = 1000
SEED      = 42

DOMAINS = {
    'synthetic': ('validation.json', 'images'),
    'lightbox':  ('test.json',       'images'),
    'sunlamp':   ('test.json',       'images'),
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
    low_freq   = power[:h//4, :w//4].mean()
    high_freq  = power[h//4:3*h//4, w//4:3*w//4].mean()
    freq_ratio = float(high_freq / (low_freq + 1e-10))

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

    return np.array([
        mean_b, std_b, overexposure, underexposure, bright_skew,
        dynamic_range, img_ent, lbp_var, sharpness, freq_ratio,
        edge, center_bright, center_contrast,
    ], dtype=np.float32)


# ── Per-Domain Extraction ─────────────────────────────────────────────────────

def extract_domain(domain: str):
    json_file, img_dir = DOMAINS[domain]
    labels = json.load(open(DATA_ROOT / domain / json_file))

    random.seed(SEED)
    sampled = random.sample(labels, min(N_SAMPLES, len(labels)))
    print(f"\n{domain}: {len(sampled)} images")

    features, filenames = [], []

    for entry in tqdm(sampled, desc=domain):
        filename = entry['filename']
        img_path = DATA_ROOT / domain / img_dir / filename
        try:
            feat = compute_stats(img_path)
            features.append(feat)
            filenames.append(filename)
        except Exception as e:
            print(f"Failed {filename}: {e}")
            continue

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