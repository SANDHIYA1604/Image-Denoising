"""
STREAMLIT APP — app.py
------------------------
Interactive web UI for the denoising system.

Run with:
  streamlit run app.py
"""

import io
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from modules.noise_detector    import detect_noise
from modules.region_classifier import classify_regions, blend_overlay
from modules.adaptive_denoiser import adaptive_denoise, baseline_denoise
from modules.noise_generator   import add_noise
from modules.evaluator         import compute_metrics, print_metrics_table


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Image Denoising System",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 14px 20px;
        margin: 6px 0;
        border-left: 4px solid #7c6af7;
    }
    .metric-label { font-size: 12px; color: #888; }
    .metric-value { font-size: 22px; font-weight: bold; color: #fff; }
    .metric-good  { color: #2ecc71; }
    .metric-bad   { color: #e74c3c; }
    .stImage img  { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def display_metric(label, value, good_threshold=None, higher_is_better=True):
    """Render a styled metric card."""
    if good_threshold is not None:
        try:
            numeric = float(str(value).replace("%","").replace(" dB",""))
            is_good = (numeric >= good_threshold) if higher_is_better else (numeric <= good_threshold)
            cls = "metric-good" if is_good else "metric-bad"
        except:
            cls = ""
    else:
        cls = ""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    st.subheader("1. Upload Image")
    uploaded_noisy = st.file_uploader("Noisy image (required)", type=["jpg", "jpeg", "png"])
    uploaded_clean = st.file_uploader("Clean reference (optional, for PSNR/SSIM)", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.subheader("2. Or Generate Test Image")
    use_synthetic = st.checkbox("Use synthetic test image")
    if use_synthetic:
        synth_noise_type = st.selectbox(
            "Noise type to add",
            ["gaussian", "salt_and_pepper", "speckle"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        synth_level = st.slider("Noise level", 5, 60, 25)

    st.markdown("---")
    st.subheader("3. Region Classifier Settings")
    texture_threshold = st.slider("Texture sensitivity", 5.0, 30.0, 12.0, 1.0,
                                   help="Lower = more pixels classified as texture")
    canny_low  = st.slider("Edge detection (low)",  10, 80,  30)
    canny_high = st.slider("Edge detection (high)", 60, 200, 100)

    st.markdown("---")
    run_btn = st.button("🚀 Run Denoising", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔬 Context-Aware Smart Image Denoising System")
st.caption("Adaptive Regional Filtering · Noise Type Detection · PSNR / SSIM Evaluation")
st.markdown("---")

if not run_btn:
    st.info("👈 Upload an image (or enable synthetic test image) and click **Run Denoising**.")
    st.markdown("""
    **How it works:**
    1. **Noise Detection** — identifies whether noise is Gaussian, Salt & Pepper, or Speckle
    2. **Region Classification** — labels every pixel as Smooth / Texture / Edge
    3. **Adaptive Denoising** — applies the best filter per region and noise type
    4. **Evaluation** — measures quality using PSNR and SSIM (if clean image provided)
    """)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Load / generate image
# ─────────────────────────────────────────────────────────────────────────────
clean_bgr = None

if use_synthetic:
    # Build a synthetic image with clear regions
    synthetic_clean = np.zeros((256, 256, 3), dtype=np.uint8)
    synthetic_clean[:128, :] = [100, 120, 80]          # smooth top
    for i in range(128, 256):                           # texture bottom
        for j in range(256):
            v = int(80 + 40 * np.sin(i * 0.5) * np.cos(j * 0.5))
            synthetic_clean[i, j] = [v, v + 10, v - 5]
    cv2.rectangle(synthetic_clean, (40, 40), (216, 216), [200, 80, 60], 4)
    cv2.circle(synthetic_clean, (128, 128), 60, [60, 160, 200], 3)

    clean_bgr  = synthetic_clean.copy()
    noisy_bgr  = add_noise(synthetic_clean, synth_noise_type,
                            sigma=synth_level, density=synth_level/500,
                            variance=synth_level/600)
    st.success(f"Synthetic image generated with {synth_noise_type.replace('_',' ').title()} noise (level={synth_level})")

elif uploaded_noisy:
    pil = Image.open(uploaded_noisy)
    noisy_bgr = pil_to_bgr(pil)
    # Resize if large
    h, w = noisy_bgr.shape[:2]
    if max(h, w) > 512:
        scale     = 512 / max(h, w)
        noisy_bgr = cv2.resize(noisy_bgr, (int(w * scale), int(h * scale)))

    if uploaded_clean:
        pil_c     = Image.open(uploaded_clean)
        clean_bgr = pil_to_bgr(pil_c)
        clean_bgr = cv2.resize(clean_bgr, (noisy_bgr.shape[1], noisy_bgr.shape[0]))
else:
    st.warning("Please upload an image or enable the synthetic test image.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🔍 Detecting noise type..."):
    noise_result = detect_noise(noisy_bgr)
    noise_type   = noise_result["noise_type"]
    noise_level  = noise_result["noise_level"]

with st.spinner("🗺️ Classifying regions..."):
    region_result = classify_regions(
        noisy_bgr,
        texture_threshold=texture_threshold,
        canny_low=canny_low,
        canny_high=canny_high,
    )
    region_map        = region_result["region_map"]
    noise_map_overlay = blend_overlay(noisy_bgr, region_result["overlay"], alpha=0.5)

with st.spinner("🧹 Applying adaptive denoising..."):
    denoise_result = adaptive_denoise(noisy_bgr, region_map, noise_type, noise_level)
    denoised       = denoise_result["denoised"]
    baseline       = baseline_denoise(noisy_bgr, noise_level)

metrics = {}
if clean_bgr is not None:
    with st.spinner("📊 Computing metrics..."):
        metrics = compute_metrics(clean_bgr, noisy_bgr, denoised, baseline)


# ─────────────────────────────────────────────────────────────────────────────
# Results: Noise Detection
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔍 Step 1 — Noise Detection")
c1, c2, c3, c4 = st.columns(4)
with c1:
    display_metric("Detected Noise Type", noise_type.replace("_", " ").title())
with c2:
    display_metric("Estimated Sigma", noise_level)
with c3:
    display_metric("Detection Confidence",
                   f"{noise_result['confidence'] * 100:.1f}%",
                   good_threshold=60)
with c4:
    scores = noise_result["scores"]
    top    = max(scores, key=scores.get)
    display_metric("Score Breakdown",
                   f"G:{scores['gaussian']:.2f}  SP:{scores['salt_and_pepper']:.2f}  Sp:{scores['speckle']:.2f}")

with st.expander("Raw statistics"):
    st.json(noise_result["stats"])


# ─────────────────────────────────────────────────────────────────────────────
# Results: Region Map
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗺️ Step 2 — Region Classification")

rc1, rc2, rc3 = st.columns(3)
with rc1:
    st.markdown("🟢 **Smooth**")
    st.metric("Coverage", f"{region_result['stats']['Smooth']}%")
with rc2:
    st.markdown("🟡 **Texture**")
    st.metric("Coverage", f"{region_result['stats']['Texture']}%")
with rc3:
    st.markdown("🔴 **Edge**")
    st.metric("Coverage", f"{region_result['stats']['Edge']}%")


# ─────────────────────────────────────────────────────────────────────────────
# Results: Images
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🖼️ Step 3 — Adaptive Denoising Results")

img_cols = st.columns(4 if clean_bgr is not None else 3)
with img_cols[0]:
    st.image(bgr_to_pil(noisy_bgr), caption="Noisy Input", use_container_width=True)
with img_cols[1]:
    st.image(bgr_to_pil(noise_map_overlay), caption="Noise Map Overlay", use_container_width=True)
with img_cols[2]:
    st.image(bgr_to_pil(denoised), caption="Adaptive Denoised Output", use_container_width=True)
if clean_bgr is not None:
    with img_cols[3]:
        st.image(bgr_to_pil(clean_bgr), caption="Clean Reference", use_container_width=True)

st.caption(f"Filters applied → Smooth: **{denoise_result['filter_used']['smooth']}**  "
           f"| Texture: **{denoise_result['filter_used']['texture']}**  "
           f"| Edge: **{denoise_result['filter_used']['edge']}**")


# ─────────────────────────────────────────────────────────────────────────────
# Results: Metrics
# ─────────────────────────────────────────────────────────────────────────────
if metrics:
    st.markdown("---")
    st.subheader("📊 Step 4 — Quality Evaluation")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        display_metric("Noisy PSNR",    f"{metrics['noisy']['psnr']} dB",    30, True)
    with m2:
        display_metric("Denoised PSNR", f"{metrics['adaptive_denoised']['psnr']} dB", 30, True)
    with m3:
        display_metric("Noisy SSIM",    f"{metrics['noisy']['ssim']}",        0.7, True)
    with m4:
        display_metric("Denoised SSIM", f"{metrics['adaptive_denoised']['ssim']}", 0.7, True)

    g1, g2 = st.columns(2)
    with g1:
        gain_psnr = metrics["improvement"]["psnr_gain"]
        color     = "normal" if gain_psnr > 0 else "inverse"
        st.metric("PSNR Improvement", f"{gain_psnr:+.2f} dB", delta=f"{gain_psnr:+.2f}", delta_color=color)
    with g2:
        gain_ssim = metrics["improvement"]["ssim_gain"]
        color     = "normal" if gain_ssim > 0 else "inverse"
        st.metric("SSIM Improvement", f"{gain_ssim:+.4f}", delta=f"{gain_ssim:+.4f}", delta_color=color)

    if "baseline_gaussian" in metrics:
        st.info(
            f"📌 Baseline (uniform Gaussian blur) → "
            f"PSNR: {metrics['baseline_gaussian']['psnr']} dB | "
            f"SSIM: {metrics['baseline_gaussian']['ssim']}  |  "
            f"Your adaptive system outperforms by "
            f"{metrics['adaptive_denoised']['psnr'] - metrics['baseline_gaussian']['psnr']:+.2f} dB PSNR"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Download denoised image
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    pil_denoised = bgr_to_pil(denoised)
    buf = io.BytesIO()
    pil_denoised.save(buf, format="PNG")
    st.download_button(
        "⬇️ Download Denoised Image",
        data=buf.getvalue(),
        file_name="denoised_output.png",
        mime="image/png",
        use_container_width=True,
    )

with col_dl2:
    pil_map = bgr_to_pil(noise_map_overlay)
    buf2 = io.BytesIO()
    pil_map.save(buf2, format="PNG")
    st.download_button(
        "⬇️ Download Noise Map",
        data=buf2.getvalue(),
        file_name="noise_map.png",
        mime="image/png",
        use_container_width=True,
    )