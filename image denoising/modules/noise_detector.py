"""
MODULE 1 — Noise Detector
--------------------------
Detects the type of noise in an image using:
  - Statistical features (mean, variance, skewness, kurtosis)
  - Pixel outlier analysis (for salt-and-pepper)
  - Frequency domain (FFT) for Gaussian noise signature
  - Variance correlation with intensity (for speckle)

Returns: noise_type (str), noise_level (float)
"""

import cv2
import numpy as np
from scipy.stats import kurtosis, skew


def detect_noise(image: np.ndarray) -> dict:
    """
    Analyzes the noise type and level in the given image.

    Args:
        image: Grayscale or BGR numpy array (uint8)

    Returns:
        dict with keys: noise_type, noise_level, confidence, stats
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_float = gray.astype(np.float32)

    # ── Step 1: Compute basic statistics ──────────────────────────────────────
    mean_val   = float(np.mean(gray_float))
    std_val    = float(np.std(gray_float))
    kurt_val   = float(kurtosis(gray_float.flatten()))
    skew_val   = float(skew(gray_float.flatten()))

    # ── Step 2: Salt-and-pepper detection ─────────────────────────────────────
    # S&P noise creates extreme pixel values at 0 and 255
    total_pixels  = gray.size
    black_pixels  = np.sum(gray == 0)
    white_pixels  = np.sum(gray == 255)
    sp_ratio      = (black_pixels + white_pixels) / total_pixels

    # ── Step 3: Speckle detection (variance vs local intensity) ───────────────
    # Speckle noise: local variance is proportional to local mean (multiplicative)
    kernel = np.ones((7, 7), np.float32) / 49
    local_mean = cv2.filter2D(gray_float, -1, kernel)
    local_sq   = cv2.filter2D(gray_float ** 2, -1, kernel)
    local_var  = local_sq - local_mean ** 2
    local_var  = np.clip(local_var, 0, None)

    # Correlation between local variance and local mean (should be high for speckle)
    valid_mask = local_mean > 10
    if np.sum(valid_mask) > 100:
        corr = float(np.corrcoef(
            local_mean[valid_mask].flatten(),
            local_var[valid_mask].flatten()
        )[0, 1])
    else:
        corr = 0.0

    # ── Step 4: Gaussian detection via FFT ────────────────────────────────────
    # Gaussian noise is spread uniformly across all frequencies
    fft         = np.fft.fft2(gray_float)
    fft_shift   = np.fft.fftshift(fft)
    magnitude   = np.log1p(np.abs(fft_shift))
    fft_std     = float(np.std(magnitude))

    # ── Step 5: Classify ──────────────────────────────────────────────────────
    scores = {
        "salt_and_pepper": 0.0,
        "gaussian":        0.0,
        "speckle":         0.0,
    }

    # S&P score: driven by outlier pixel ratio and high kurtosis
    scores["salt_and_pepper"] = min(1.0, sp_ratio * 20 + max(0, kurt_val - 3) * 0.05)

    # Speckle score: driven by variance-mean correlation
    scores["speckle"] = max(0.0, corr) * (1 - scores["salt_and_pepper"])

    # Gaussian score: residual (assumes one dominant noise type)
    scores["gaussian"] = max(0.0, 1.0 - scores["salt_and_pepper"] - scores["speckle"])
    # Boost gaussian if FFT is uniform (flat spectrum = white noise)
    if fft_std < 3.0:
        scores["gaussian"] = min(1.0, scores["gaussian"] + 0.3)

    # Normalize scores
    total = sum(scores.values()) or 1.0
    scores = {k: v / total for k, v in scores.items()}

    noise_type  = max(scores, key=scores.get)
    confidence  = scores[noise_type]

    # Noise level: estimate sigma from median absolute deviation (robust)
    noise_level = float(np.median(np.abs(gray_float - np.median(gray_float))) / 0.6745)

    return {
        "noise_type":  noise_type,
        "noise_level": round(noise_level, 2),
        "confidence":  round(confidence, 3),
        "scores":      {k: round(v, 3) for k, v in scores.items()},
        "stats": {
            "mean":      round(mean_val, 2),
            "std":       round(std_val, 2),
            "kurtosis":  round(kurt_val, 3),
            "skewness":  round(skew_val, 3),
            "sp_ratio":  round(sp_ratio, 4),
            "speckle_corr": round(corr, 3),
        },
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Generate a synthetic noisy image for testing
    clean = np.random.randint(80, 180, (256, 256), dtype=np.uint8)

    # Add Gaussian noise
    noise  = np.random.normal(0, 25, clean.shape).astype(np.float32)
    noisy  = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    result = detect_noise(noisy)
    print("=== Noise Detection Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")