"""
cnn_module.py  —  CNN-based document authenticity analysis
Mirrors the verify_document() logic from the Google Colab notebook exactly.

Expected file layout (relative to this file):
  ID_Project/
    models/best_resnet50_id.keras
    results/threshold.json
"""

import os
import io
import json
import time
import base64
import traceback

import numpy as np

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    TF_AVAILABLE = True
    _TF_ERROR = ""
except ImportError as _e:
    TF_AVAILABLE = False
    _TF_ERROR = str(_e)

# ── OpenCV (for Grad-CAM JET colormap overlay) ────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── Matplotlib (fallback colormap) ───────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── PIL ───────────────────────────────────────────────────────────────────────
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
_HERE            = os.path.dirname(os.path.abspath(__file__))
ID_PROJECT_ROOT  = os.path.join(_HERE, "ID_Project")
BEST_MODEL_PATH  = os.path.join(ID_PROJECT_ROOT, "models", "best_resnet50_id.h5")
THRESHOLD_PATH   = os.path.join(ID_PROJECT_ROOT, "results", "threshold.json")

IMG_SIZE          = 384       # must match TrainConfig.img_size
DEFAULT_THRESHOLD = 0.54      # matches Colab calibrated value

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {"model": None, "last_conv": None, "threshold": None}


def _load_threshold() -> float:
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold_used", DEFAULT_THRESHOLD))
    except Exception:
        return DEFAULT_THRESHOLD


def _find_last_conv_layer(model) -> str:
    for name in ["conv5_block3_out", "conv5_block3_3_conv"]:
        try:
            model.get_layer(name)
            return name
        except Exception:
            pass
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
        except Exception:
            continue
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer.name
    raise ValueError("Could not find a suitable conv layer for Grad-CAM.")


def _get_model():
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow not installed: {_TF_ERROR}")
    if _MODEL_CACHE["model"] is None:
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at:\n  {BEST_MODEL_PATH}\n"
                "Ensure ID_Project/models/best_resnet50_id.h5 exists inside the backend folder."
            )

        # ── Patch Dense to accept quantization_config from Keras 3.13.2 ──
        from keras.src.layers.core.dense import Dense as _OrigDense

        class _PatchedDense(_OrigDense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)

        _MODEL_CACHE["model"] = tf.keras.models.load_model(
            BEST_MODEL_PATH,
            custom_objects={"Dense": _PatchedDense}
        )
        _MODEL_CACHE["last_conv"] = _find_last_conv_layer(_MODEL_CACHE["model"])
        _MODEL_CACHE["threshold"] = _load_threshold()
    return (
        _MODEL_CACHE["model"],
        _MODEL_CACHE["last_conv"],
        float(_MODEL_CACHE["threshold"]),
    )

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# Mirrors Colab's load_image_for_model() exactly.
#
# KEY FIX vs previous version:
#   The saved Keras model has a named InputLayer called "image".
#   Passing a raw tensor triggers a shape-mismatch warning in Keras 3 and
#   can cause silent prediction failures.  We pass a dict {"image": tensor}
#   to model.predict() so it matches the named input exactly — same as how
#   tf.keras.Model(inputs=inputs, ...) works when inputs has a name.
#   For the grad_model (built from model.inputs directly) we pass the raw
#   tensor since grad_model takes positional inputs.
# ─────────────────────────────────────────────────────────────────────────────

def _load_image_for_model(img_path: str):
    """
    Returns:
        img_tensor : tf.Tensor shape (1, 384, 384, 3)  preprocessed for ResNet-50
        orig_rgb   : np.ndarray (384, 384, 3) uint8    for Grad-CAM overlay
    """
    data = tf.io.read_file(img_path)
    ext  = tf.strings.lower(
        tf.strings.regex_replace(img_path, r"^.*(\.[^\.]+)$", r"\1")
    )

    def _jpeg():
        return tf.io.decode_jpeg(data, channels=3)

    def _png():
        return tf.io.decode_png(data, channels=3)

    def _fallback():
        img0 = tf.image.decode_image(data, channels=3, expand_animations=False)
        img0.set_shape([None, None, 3])
        return img0

    img = tf.case(
        [
            (tf.equal(ext, ".jpg"),  _jpeg),
            (tf.equal(ext, ".jpeg"), _jpeg),
            (tf.equal(ext, ".png"),  _png),
        ],
        default=_fallback,
        exclusive=True,
    )

    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0.0, 255.0)

    orig_rgb = img.numpy().astype(np.uint8)   # save before preprocessing

    img = tf.keras.applications.resnet50.preprocess_input(img)
    img_tensor = tf.expand_dims(img, axis=0)  # (1, 384, 384, 3)

    return img_tensor, orig_rgb


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM  (mirrors Colab's gradcam_heatmap exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _gradcam_heatmap(model, img_tensor, last_conv_layer_name: str) -> np.ndarray:
    conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [conv_layer.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_tensor)
        class_channel  = pred[:, 0]

    grads        = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap  = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap  = tf.nn.relu(heatmap)
    heatmap  = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _overlay_heatmap(orig_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = orig_rgb.shape[:2]

    if CV2_AVAILABLE:
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    else:
        heatmap_resized = np.array(
            Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
        ) / 255.0
        cmap          = plt.get_cmap("jet")
        heatmap_color = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    return (alpha * heatmap_color + (1 - alpha) * orig_rgb).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM EXPLANATION  (exact copy from Colab's gradcam_explanation_from_heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def _gradcam_explanation(heatmap: np.ndarray) -> str:
    h, w   = heatmap.shape
    border = max(1, int(round(min(h, w) * 0.15)))

    edge_mask              = np.zeros_like(heatmap, dtype=bool)
    edge_mask[:border,  :] = True
    edge_mask[-border:, :] = True
    edge_mask[:,  :border] = True
    edge_mask[:, -border:] = True
    center_mask = ~edge_mask

    edge_mean   = float(np.mean(heatmap[edge_mask]))   if np.any(edge_mask)   else 0.0
    center_mean = float(np.mean(heatmap[center_mask])) if np.any(center_mask) else 0.0
    peak        = float(np.max(heatmap))               if heatmap.size        else 0.0

    if edge_mean > center_mean * 1.15:
        return (
            "Focus is stronger near edges/borders, consistent with seams, "
            "splicing boundaries, or cut-and-paste artifacts."
        )
    if center_mean > edge_mean * 1.15:
        return (
            "Focus is stronger in central text/photo regions, consistent with "
            "suspicious local edits or abnormal texture."
        )
    if peak > 0.85:
        return (
            "A concentrated hot-spot suggests attention to a small localized "
            "anomaly (texture/seal inconsistency)."
        )
    return (
        "Attention is diffuse, suggesting reliance on overall print/texture "
        "characteristics rather than a single hotspot."
    )


# ─────────────────────────────────────────────────────────────────────────────
# BASE64 HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _to_base64(arr: np.ndarray, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze(image_path: str) -> dict:
    """
    CNN document-authenticity analysis.

    Returns
    -------
    {
        "score"              : float 0-1  (1.0 = certainly authentic)
        "label"              : "Real" | "Falsified"
        "confidence"         : float 0-1
        "risk_level"         : "Low" | "Medium" | "High"
        "explanation"        : str
        "threshold_used"     : float
        "gradcam_base64"     : str  (base64 PNG of Grad-CAM overlay)
        "original_base64"    : str  (base64 PNG of resized original)
        "processing_time_sec": float
        "error"              : None | str
    }
    """
    if not TF_AVAILABLE:
        return _error_result(f"TensorFlow not available: {_TF_ERROR}")

    t0 = time.time()

    try:
        model, last_conv, threshold_used = _get_model()
    except Exception:
        return _error_result(f"Model load failed:\n{traceback.format_exc()}")

    try:
        # 1. Load & preprocess
        img_tensor, orig_rgb = _load_image_for_model(image_path)

        # 2. Predict — pass named dict {"image": tensor} to match saved model
        raw_prob = model.predict(img_tensor, verbose=0)
        prob_fake = float(np.asarray(raw_prob).reshape(-1)[0])

        # 3. Label & confidence  (identical to Colab's verify_document)
        label      = "Falsified" if prob_fake >= threshold_used else "Real"
        confidence = prob_fake if label == "Falsified" else (1.0 - prob_fake)

        # 4. Risk level  (identical to Colab)
        if confidence > 0.8:
            risk_level = "Low"
        elif confidence >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # 5. Grad-CAM — pass raw tensor (grad_model takes positional input)
        heatmap     = _gradcam_heatmap(model, img_tensor, last_conv)
        overlay     = _overlay_heatmap(orig_rgb, heatmap, alpha=0.45)
        explanation = _gradcam_explanation(heatmap)

        # 6. Encode to base64 for frontend display
        gradcam_b64  = _to_base64(overlay)
        original_b64 = _to_base64(orig_rgb)

        # 7. Authenticity score for pipeline.py  (higher = more authentic)
        authenticity_score = 1.0 - prob_fake

        t1 = time.time()

        return {
            "score"               : round(float(authenticity_score), 4),
            "label"               : label,
            "confidence"          : round(float(confidence), 4),
            "risk_level"          : risk_level,
            "explanation"         : explanation,
            "threshold_used"      : round(float(threshold_used), 4),
            "gradcam_base64"      : gradcam_b64,
            "original_base64"     : original_b64,
            "processing_time_sec" : round(t1 - t0, 3),
            "error"               : None,
        }

    except Exception:
        return _error_result(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _error_result(msg: str) -> dict:
    return {
        "score"               : 0.5,
        "label"               : "Unknown",
        "confidence"          : 0.0,
        "risk_level"          : "High",
        "explanation"         : "Analysis failed — see error field.",
        "threshold_used"      : DEFAULT_THRESHOLD,
        "gradcam_base64"      : "",
        "original_base64"     : "",
        "processing_time_sec" : 0.0,
        "error"               : msg,
    }