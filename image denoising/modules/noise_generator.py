"""
UTILITY — Noise Generator
---------------------------
Adds synthetic noise to a clean image for testing.
Use this when you only have one image (no clean reference).

Noise types:
  - Gaussian    : additive normally-distributed noise
  - Salt & Pepper: random black/white pixel spikes
  - Speckle     : multiplicative noise (common in ultrasound/radar)
"""

import cv2
import numpy as np


def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Adds zero-mean Gaussian noise.
    sigma: standard deviation (5=light, 25=medium, 50=heavy)
    """
    noise  = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy  = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper(image: np.ndarray, density: float = 0.05) -> np.ndarray:
    """
    Adds salt-and-pepper (impulse) noise.
    density: fraction of pixels affected (0.05 = 5%)
    """
    noisy  = image.copy()
    h, w   = image.shape[:2]
    n      = int(density * h * w)

    # Salt (white pixels)
    salt_y = np.random.randint(0, h, n)
    salt_x = np.random.randint(0, w, n)
    if len(image.shape) == 3:
        noisy[salt_y, salt_x] = [255, 255, 255]
    else:
        noisy[salt_y, salt_x] = 255

    # Pepper (black pixels)
    pepper_y = np.random.randint(0, h, n)
    pepper_x = np.random.randint(0, w, n)
    if len(image.shape) == 3:
        noisy[pepper_y, pepper_x] = [0, 0, 0]
    else:
        noisy[pepper_y, pepper_x] = 0

    return noisy


def add_speckle_noise(image: np.ndarray, variance: float = 0.04) -> np.ndarray:
    """
    Adds multiplicative speckle noise: output = image + image * noise
    variance: noise variance (0.02=light, 0.1=heavy)
    """
    noise  = np.random.randn(*image.shape).astype(np.float32)
    noisy  = image.astype(np.float32) + image.astype(np.float32) * noise * np.sqrt(variance)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_noise(image: np.ndarray, noise_type: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Convenience wrapper. noise_type: 'gaussian' | 'salt_and_pepper' | 'speckle'
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(image, sigma=kwargs.get("sigma", 25.0))
    elif noise_type == "salt_and_pepper":
        return add_salt_and_pepper(image, density=kwargs.get("density", 0.05))
    elif noise_type == "speckle":
        return add_speckle_noise(image, variance=kwargs.get("variance", 0.04))
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img = cv2.imread("data/clean/test.jpg")
    if img is None:
        print("Creating synthetic test image...")
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        cv2.rectangle(img, (60, 60), (196, 196), (200, 100, 50), -1)

    gauss = add_noise(img, "gaussian",        sigma=25)
    sp    = add_noise(img, "salt_and_pepper",  density=0.05)
    speck = add_noise(img, "speckle",          variance=0.04)

    cv2.imwrite("data/noisy/gaussian_noisy.jpg", gauss)
    cv2.imwrite("data/noisy/sp_noisy.jpg",       sp)
    cv2.imwrite("data/noisy/speckle_noisy.jpg",  speck)
    print("Noisy images saved to data/noisy/")