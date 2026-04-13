"""
MODULE 2 — Region Classifier
------------------------------
Segments an image into three region types:
  0 = Smooth   (flat areas — e.g. sky, walls)
  1 = Texture  (detailed areas — e.g. grass, fabric)
  2 = Edge     (boundary areas — e.g. object contours)

Uses:
  - Canny edge detection for edge regions
  - Local standard deviation for smooth vs texture classification

Returns:
  region_map  — same H×W as input, values in {0, 1, 2}
  overlay     — BGR color-coded visualization image
"""

import cv2
import numpy as np


# ── Color palette for the noise map ──────────────────────────────────────────
REGION_COLORS = {
    0: (180, 220, 100),   # Smooth  → soft green  (BGR)
    1: (50,  180, 230),   # Texture → amber/orange (BGR)
    2: (60,   60, 220),   # Edge    → red          (BGR)
}

REGION_LABELS = {0: "Smooth", 1: "Texture", 2: "Edge"}


def classify_regions(image: np.ndarray, 
                     block_size: int = 8,
                     canny_low: int = 30,
                     canny_high: int = 100,
                     texture_threshold: float = 12.0) -> dict:
    """
    Classifies every pixel into Smooth / Texture / Edge.

    Args:
        image:             Grayscale or BGR numpy array
        block_size:        Size of local window for std-dev computation
        canny_low:         Lower threshold for Canny edge detector
        canny_high:        Upper threshold for Canny edge detector
        texture_threshold: Local std-dev above this → texture; below → smooth

    Returns:
        dict with:
            region_map   : H×W uint8 array (0=smooth, 1=texture, 2=edge)
            overlay      : H×W×3 BGR color visualization
            stats        : percentage of each region type
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # ── Step 1: Detect edges using Canny ─────────────────────────────────────
    blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_map   = cv2.Canny(blurred, canny_low, canny_high)

    # Dilate edges slightly so the edge region has thickness
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_dilated = cv2.dilate(edge_map, kernel, iterations=1)

    # ── Step 2: Compute local standard deviation ──────────────────────────────
    gray_f     = gray.astype(np.float32)
    # local mean via box filter
    local_mean = cv2.blur(gray_f, (block_size, block_size))
    local_sq   = cv2.blur(gray_f ** 2, (block_size, block_size))
    local_var  = np.clip(local_sq - local_mean ** 2, 0, None)
    local_std  = np.sqrt(local_var)

    # ── Step 3: Build region map ──────────────────────────────────────────────
    region_map = np.zeros((h, w), dtype=np.uint8)  # default = smooth (0)

    # Texture: high local variance, NOT an edge
    texture_mask = (local_std >= texture_threshold) & (edge_dilated == 0)
    region_map[texture_mask] = 1

    # Edge: wherever Canny fired (overrides texture)
    region_map[edge_dilated > 0] = 2

    # ── Step 4: Build color overlay ───────────────────────────────────────────
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for region_id, color in REGION_COLORS.items():
        mask = region_map == region_id
        overlay[mask] = color

    # ── Step 5: Compute statistics ───────────────────────────────────────────
    total = h * w
    stats = {
        REGION_LABELS[i]: round(float(np.sum(region_map == i)) / total * 100, 1)
        for i in range(3)
    }

    return {
        "region_map": region_map,
        "overlay":    overlay,
        "stats":      stats,
        "edge_map":   edge_map,
        "local_std":  local_std,
    }


def blend_overlay(image: np.ndarray, overlay: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blends the color overlay on top of the original image.

    Args:
        image:   Original BGR image
        overlay: Color-coded region map (BGR)
        alpha:   Overlay opacity (0 = original only, 1 = overlay only)

    Returns:
        Blended BGR image
    """
    if len(image.shape) == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    blended = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    return blended


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create a synthetic test image with distinct regions
    img = np.zeros((256, 256), dtype=np.uint8)
    img[0:128, :]   = 120                         # smooth top half
    img[128:, :]    = np.random.randint(80, 160, (128, 256), dtype=np.uint8)  # texture bottom
    cv2.rectangle(img, (60, 60), (196, 196), 255, 4)  # edge rectangle

    result = classify_regions(img)
    print("=== Region Classification ===")
    print("Region stats:", result["stats"])
    print("Region map shape:", result["region_map"].shape)
    print("Unique values in map:", np.unique(result["region_map"]))