"""
MAIN PIPELINE — run_pipeline.py
---------------------------------
Runs the full denoising pipeline on one image:
  1. Load image
  2. Detect noise type
  3. Classify regions
  4. Adaptive denoise
  5. Evaluate (PSNR + SSIM)
  6. Save results

Usage:
  python run_pipeline.py --input data/noisy/gaussian_noisy.jpg
  python run_pipeline.py --input data/noisy/gaussian_noisy.jpg --clean data/clean/original.jpg
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.noise_detector    import detect_noise
from modules.region_classifier import classify_regions, blend_overlay
from modules.adaptive_denoiser import adaptive_denoise, baseline_denoise
from modules.evaluator         import compute_metrics, generate_comparison_plot, print_metrics_table


def run_pipeline(input_path: str, clean_path: str = None, save_dir: str = "data/output"):
    """
    Full pipeline: load → detect → classify → denoise → evaluate → save.

    Args:
        input_path: Path to noisy input image
        clean_path: Path to clean reference (optional, needed for metrics)
        save_dir:   Directory to save output files
    """
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # ── STEP 1: Load image ────────────────────────────────────────────────────
    print(f"\n[1/5] Loading image: {input_path}")
    noisy = cv2.imread(input_path)
    if noisy is None:
        raise FileNotFoundError(f"Cannot open image: {input_path}")

    # Resize if too large (keep aspect ratio, max 512px on longest side)
    h, w = noisy.shape[:2]
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        noisy = cv2.resize(noisy, (int(w * scale), int(h * scale)))
        print(f"  Resized to {noisy.shape[1]}×{noisy.shape[0]}")

    clean = None
    if clean_path:
        clean = cv2.imread(clean_path)
        if clean is not None:
            clean = cv2.resize(clean, (noisy.shape[1], noisy.shape[0]))
            print(f"  Loaded clean reference: {clean_path}")

    # ── STEP 2: Detect noise ──────────────────────────────────────────────────
    print("\n[2/5] Detecting noise type...")
    noise_result = detect_noise(noisy)
    noise_type   = noise_result["noise_type"]
    noise_level  = noise_result["noise_level"]
    print(f"  Type    : {noise_type.replace('_',' ').title()}")
    print(f"  Level   : sigma ≈ {noise_level}")
    print(f"  Scores  : {noise_result['scores']}")

    # ── STEP 3: Classify regions ──────────────────────────────────────────────
    print("\n[3/5] Classifying image regions...")
    region_result = classify_regions(noisy)
    region_map    = region_result["region_map"]
    region_stats  = region_result["stats"]
    print(f"  Region breakdown: {region_stats}")

    noise_map_overlay = blend_overlay(noisy, region_result["overlay"], alpha=0.5)
    noise_map_path    = os.path.join(save_dir, f"{base_name}_noise_map.jpg")
    cv2.imwrite(noise_map_path, noise_map_overlay)
    print(f"  Noise map saved → {noise_map_path}")

    # ── STEP 4: Adaptive denoising ────────────────────────────────────────────
    print("\n[4/5] Running adaptive denoising...")
    denoise_result = adaptive_denoise(noisy, region_map, noise_type, noise_level)
    denoised       = denoise_result["denoised"]
    print(f"  Filters used: {denoise_result['filter_used']}")

    denoised_path = os.path.join(save_dir, f"{base_name}_denoised.jpg")
    cv2.imwrite(denoised_path, denoised)
    print(f"  Denoised image saved → {denoised_path}")

    # ── STEP 5: Evaluate ──────────────────────────────────────────────────────
    print("\n[5/5] Evaluating quality metrics...")
    if clean is not None:
        baseline = baseline_denoise(noisy, noise_level)
        metrics  = compute_metrics(clean, noisy, denoised, baseline)
        print_metrics_table(metrics)

        # Save comparison plot
        fig = generate_comparison_plot(
            original=clean,
            noisy=noisy,
            denoised=denoised,
            noise_map_overlay=noise_map_overlay,
            metrics=metrics,
            noise_type=noise_type,
            save_path=os.path.join(save_dir, f"{base_name}_comparison.png"),
        )
        plt.show()
    else:
        print("  No clean reference provided — skipping PSNR/SSIM.")
        print("  Tip: provide --clean <path> to enable quality metrics.")
        metrics = {}

    print(f"\n Done! All outputs saved in: {save_dir}/\n")
    return denoised, region_map, noise_result, metrics


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context-Aware Image Denoising Pipeline")
    parser.add_argument("--input", required=True, help="Path to noisy image")
    parser.add_argument("--clean", default=None,  help="Path to clean reference (for metrics)")
    parser.add_argument("--out",   default="data/output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.input, args.clean, args.out)