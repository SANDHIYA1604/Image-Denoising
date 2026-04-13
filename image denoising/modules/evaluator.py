"""
MODULE 4 — Evaluator
----------------------
Computes image quality metrics:
  - PSNR  (Peak Signal-to-Noise Ratio)   — higher is better, >30 dB is good
  - SSIM  (Structural Similarity Index)  — higher is better, 1.0 = perfect

Also generates a side-by-side comparison plot.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


def compute_metrics(original: np.ndarray,
                    noisy: np.ndarray,
                    denoised: np.ndarray,
                    baseline: np.ndarray = None) -> dict:
    """
    Computes PSNR and SSIM for noisy vs denoised images,
    both compared against the clean original.

    Args:
        original:  Clean reference image
        noisy:     Noisy version
        denoised:  Output of adaptive denoiser
        baseline:  Output of simple Gaussian (for comparison)

    Returns:
        dict with metric values
    """
    is_color = len(original.shape) == 3

    def _psnr(ref, img):
        return round(float(psnr_fn(ref, img, data_range=255)), 2)

    def _ssim(ref, img):
        if is_color:
            return round(float(ssim_fn(ref, img,
                                       data_range=255,
                                       channel_axis=2)), 4)
        else:
            return round(float(ssim_fn(ref, img, data_range=255)), 4)

    results = {
        "noisy": {
            "psnr": _psnr(original, noisy),
            "ssim": _ssim(original, noisy),
        },
        "adaptive_denoised": {
            "psnr": _psnr(original, denoised),
            "ssim": _ssim(original, denoised),
        },
    }

    if baseline is not None:
        results["baseline_gaussian"] = {
            "psnr": _psnr(original, baseline),
            "ssim": _ssim(original, baseline),
        }

    # Compute improvement
    results["improvement"] = {
        "psnr_gain": round(
            results["adaptive_denoised"]["psnr"] - results["noisy"]["psnr"], 2),
        "ssim_gain": round(
            results["adaptive_denoised"]["ssim"] - results["noisy"]["ssim"], 4),
    }

    return results


def generate_comparison_plot(original: np.ndarray,
                              noisy: np.ndarray,
                              denoised: np.ndarray,
                              noise_map_overlay: np.ndarray,
                              metrics: dict,
                              noise_type: str,
                              save_path: str = None) -> plt.Figure:
    """
    Generates a 4-panel comparison figure:
    [Original] [Noisy] [Noise Map] [Denoised]

    Args:
        original:          Clean reference image
        noisy:             Noisy input
        denoised:          Adaptive denoiser output
        noise_map_overlay: Color-coded region map blended on image
        metrics:           Output of compute_metrics()
        noise_type:        Detected noise type string
        save_path:         If provided, saves the figure to this path

    Returns:
        matplotlib Figure object
    """

    def _to_rgb(img):
        """Convert BGR or grayscale to RGB for matplotlib."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    fig = plt.figure(figsize=(16, 5), facecolor="#0f1117")
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.06, hspace=0)

    panels = [
        (_to_rgb(original),          "Original (Clean)",
         None),
        (_to_rgb(noisy),             f"Noisy Input\nPSNR: {metrics['noisy']['psnr']} dB | SSIM: {metrics['noisy']['ssim']}",
         "#e74c3c"),
        (_to_rgb(noise_map_overlay), f"Noise Map\nDetected: {noise_type.replace('_', ' ').title()}",
         "#f39c12"),
        (_to_rgb(denoised),          f"Adaptive Denoised\nPSNR: {metrics['adaptive_denoised']['psnr']} dB | SSIM: {metrics['adaptive_denoised']['ssim']}",
         "#2ecc71"),
    ]

    for i, (img, title, accent) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#0f1117")

        # Colored border for noisy / noise map / denoised panels
        if accent:
            for spine in ax.spines.values():
                spine.set_edgecolor(accent)
                spine.set_linewidth(2.5)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
                spine.set_linewidth(1)

        color = accent if accent else "#aaaaaa"
        ax.set_title(title, color=color, fontsize=9.5,
                     pad=8, fontweight="bold", linespacing=1.6)

    # Bottom annotation
    gain_psnr = metrics["improvement"]["psnr_gain"]
    gain_ssim = metrics["improvement"]["ssim_gain"]
    fig.text(
        0.5, 0.01,
        f"PSNR improvement: +{gain_psnr} dB   |   SSIM improvement: +{gain_ssim}",
        ha="center", va="bottom",
        color="#7f8c8d", fontsize=9,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0f1117")
        print(f"Saved comparison plot → {save_path}")

    return fig


def print_metrics_table(metrics: dict):
    """Pretty-prints the metrics to terminal."""
    print("\n" + "=" * 50)
    print("  IMAGE QUALITY EVALUATION")
    print("=" * 50)
    for key, vals in metrics.items():
        if key == "improvement":
            continue
        print(f"\n  {key.replace('_', ' ').upper()}")
        if isinstance(vals, dict):
            for metric, val in vals.items():
                print(f"    {metric.upper():10s}: {val}")
    print(f"\n  IMPROVEMENT OVER NOISY INPUT")
    print(f"    PSNR gain : +{metrics['improvement']['psnr_gain']} dB")
    print(f"    SSIM gain : +{metrics['improvement']['ssim_gain']}")
    print("=" * 50 + "\n")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    original = np.random.randint(80, 180, (128, 128, 3), dtype=np.uint8)
    noise    = np.random.normal(0, 25, original.shape).astype(np.float32)
    noisy    = np.clip(original.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    denoised = cv2.GaussianBlur(noisy, (5, 5), 1.5)

    metrics = compute_metrics(original, noisy, denoised)
    print_metrics_table(metrics)