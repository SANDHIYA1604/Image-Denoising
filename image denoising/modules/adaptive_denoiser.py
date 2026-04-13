"""
MODULE 3 — Adaptive Denoising Engine
--------------------------------------
Applies region-aware, noise-type-aware denoising.

Strategy matrix:
  ┌──────────────────┬─────────────┬────────────────────┬──────────────────┐
  │ Region \\ Noise   │  Gaussian   │  Salt & Pepper     │  Speckle         │
  ├──────────────────┼─────────────┼────────────────────┼──────────────────┤
  │ Smooth           │ Gaussian    │ Median             │ Bilateral        │
  │ Texture          │ NL-Means    │ Median → NL-Means  │ NL-Means         │
  │ Edge             │ Bilateral   │ Median → Bilateral │ Bilateral        │
  └──────────────────┴─────────────┴────────────────────┴──────────────────┘

Returns: denoised image (BGR or grayscale, same dtype as input)
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Individual filter functions
# ─────────────────────────────────────────────────────────────────────────────

def _apply_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur — good for smooth regions with Gaussian noise."""
    # Kernel size must be odd; derive from sigma
    ksize = int(2 * round(3 * sigma) + 1)
    ksize = max(ksize, 3)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def _apply_median(image: np.ndarray, noise_level: float) -> np.ndarray:
    """Median filter — optimal for salt-and-pepper noise."""
    # Stronger noise → bigger kernel (must be odd)
    ksize = 3 if noise_level < 20 else 5
    return cv2.medianBlur(image, ksize)


def _apply_bilateral(image: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Bilateral filter — preserves edges by only averaging pixels
    that are spatially close AND have similar intensity.
    """
    d         = 9           # diameter of pixel neighbourhood
    sigma_col = noise_level * 2.5   # larger → more color mixing
    sigma_sp  = 75          # spatial extent
    return cv2.bilateralFilter(image, d, sigma_col, sigma_sp)


def _apply_nlmeans(image: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Non-Local Means — searches whole image for similar patches.
    Best for texture regions; computationally heavier.
    """
    h_param  = max(3, min(noise_level * 1.2, 30))  # filter strength
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=h_param,
            hColor=h_param,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    else:
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=h_param,
            templateWindowSize=7,
            searchWindowSize=21,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main denoising function
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_denoise(image: np.ndarray,
                     region_map: np.ndarray,
                     noise_type: str,
                     noise_level: float) -> dict:
    """
    Applies region-aware denoising based on detected noise type.

    Args:
        image:       Original noisy image (BGR or grayscale, uint8)
        region_map:  H×W array with values {0=smooth, 1=texture, 2=edge}
        noise_type:  One of 'gaussian', 'salt_and_pepper', 'speckle'
        noise_level: Estimated sigma from noise detection

    Returns:
        dict with:
            denoised       : Final denoised image (same shape/dtype as input)
            smooth_result  : Filter applied to smooth regions only
            texture_result : Filter applied to texture regions only
            edge_result    : Filter applied to edge regions only
            filter_used    : dict mapping region → filter name
    """
    is_color = len(image.shape) == 3
    h, w     = image.shape[:2]

    # ── Determine filter per region based on noise type ───────────────────────
    filter_map = _get_filter_strategy(noise_type)

    # ── Apply each filter to the full image ───────────────────────────────────
    # (We apply each filter to the entire image, then blend using masks)
    results = {}
    sigma   = max(noise_level, 5.0)

    for region_id, filter_name in filter_map.items():
        results[region_id] = _run_filter(image, filter_name, sigma)

    # ── Blend results using region masks ──────────────────────────────────────
    if is_color:
        denoised = np.zeros_like(image, dtype=np.float32)
        weight   = np.zeros((h, w), dtype=np.float32)
    else:
        denoised = np.zeros((h, w), dtype=np.float32)
        weight   = np.zeros((h, w), dtype=np.float32)

    for region_id, filtered in results.items():
        mask = (region_map == region_id).astype(np.float32)
        if is_color:
            denoised += filtered.astype(np.float32) * mask[:, :, np.newaxis]
        else:
            denoised += filtered.astype(np.float32) * mask
        weight += mask

    # Avoid divide-by-zero (shouldn't happen but safety net)
    weight = np.maximum(weight, 1e-8)
    if is_color:
        denoised /= weight[:, :, np.newaxis]
    else:
        denoised /= weight

    denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    return {
        "denoised":       denoised,
        "smooth_result":  results.get(0, image),
        "texture_result": results.get(1, image),
        "edge_result":    results.get(2, image),
        "filter_used":    {
            "smooth":  filter_map[0],
            "texture": filter_map[1],
            "edge":    filter_map[2],
        },
    }


def _get_filter_strategy(noise_type: str) -> dict:
    """Returns {region_id: filter_name} mapping based on noise type."""
    strategies = {
        "gaussian": {
            0: "gaussian",   # smooth  → Gaussian blur
            1: "nlmeans",    # texture → NL-Means
            2: "bilateral",  # edge    → Bilateral
        },
        "salt_and_pepper": {
            0: "median",     # smooth  → Median
            1: "median",     # texture → Median
            2: "median",     # edge    → Median (best for S&P everywhere)
        },
        "speckle": {
            0: "bilateral",  # smooth  → Bilateral
            1: "nlmeans",    # texture → NL-Means
            2: "bilateral",  # edge    → Bilateral
        },
    }
    return strategies.get(noise_type, strategies["gaussian"])


def _run_filter(image: np.ndarray, filter_name: str, sigma: float) -> np.ndarray:
    """Dispatch to the correct filter function."""
    if filter_name == "gaussian":
        return _apply_gaussian(image, sigma)
    elif filter_name == "median":
        return _apply_median(image, sigma)
    elif filter_name == "bilateral":
        return _apply_bilateral(image, sigma)
    elif filter_name == "nlmeans":
        return _apply_nlmeans(image, sigma)
    else:
        return image.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparison (uniform filter — to show adaptive is better)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_denoise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """Simple uniform Gaussian blur — used for comparison in evaluation."""
    return _apply_gaussian(image, max(noise_level * 0.8, 3.0))


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic test
    clean = np.random.randint(80, 180, (128, 128, 3), dtype=np.uint8)
    noise = np.random.normal(0, 20, clean.shape).astype(np.float32)
    noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Fake region map
    region_map = np.zeros((128, 128), dtype=np.uint8)
    region_map[64:, :] = 1
    region_map[30:35, :] = 2

    result = adaptive_denoise(noisy, region_map, "gaussian", 20.0)
    print("Denoising complete!")
    print("Output shape:", result["denoised"].shape)
    print("Filters used:", result["filter_used"])