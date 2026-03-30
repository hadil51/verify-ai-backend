import os
import io
import base64
import tempfile
import numpy as np
from PIL import Image, ExifTags

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

try:
    import jpegio
    JPEGIO_AVAILABLE = True
except ImportError:
    JPEGIO_AVAILABLE = False


# ─────────────────────────────────────────────
# HELPERS — chargement
# ─────────────────────────────────────────────

def _load_rgb(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32)


def _to_jpeg_bytes(image_path: str, quality: int) -> bytes:
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _to_temp_jpeg(image_path: str) -> tuple[str, bool]:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return image_path, False
    img = Image.open(image_path).convert("RGB")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG", quality=95)
    tmp.close()
    return tmp.name, True


# ─────────────────────────────────────────────
# HELPER — jet colormap (bleu→vert→rouge) comme Grad-CAM
# ─────────────────────────────────────────────

def _apply_jet_colormap(gray_map: np.ndarray) -> np.ndarray:
    """
    gray_map : 2D float array, valeurs quelconques.
    Retourne un array H×W×3 uint8 en colormap jet (bleu→vert→rouge).
    """
    mn, mx = gray_map.min(), gray_map.max()
    if mx - mn < 1e-8:
        normed = np.zeros_like(gray_map)
    else:
        normed = (gray_map - mn) / (mx - mn)   # 0..1

    # Jet colormap manuelle
    r = np.clip(1.5 - np.abs(normed - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(normed - 0.50) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(normed - 0.25) * 4, 0, 1)

    jet = np.stack([r, g, b], axis=-1)           # H×W×3 float [0,1]
    return (jet * 255).astype(np.uint8)


def _blend_heatmap(original_rgb: np.ndarray, heatmap_rgb: np.ndarray,
                   alpha: float = 0.55) -> str:
    """
    Blende heatmap (H×W×3 uint8) sur l'image originale redimensionnée.
    Retourne un string base64 PNG.
    """
    h, w = original_rgb.shape[:2]
    orig_u8 = original_rgb.astype(np.uint8)

    # Redimensionne la heatmap à la taille de l'original
    heat_img = Image.fromarray(heatmap_rgb).resize((w, h), Image.BILINEAR)
    heat_arr = np.array(heat_img, dtype=np.float32)

    # Blend
    blended = (1 - alpha) * orig_u8.astype(np.float32) + alpha * heat_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────
# MODULE 1 — Analyse EXIF (incohérences internes)
# ─────────────────────────────────────────────

SOFTWARE_EDITORS = [
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
    "paint.net", "snapseed", "facetune", "meitu", "picsart",
    "canva", "fotor", "pixlr", "capture one", "darktable",
    "corel", "inkscape", "illustrator"
]


def _analyze_exif(image_path: str) -> dict:
    """
    Analyse les incohérences dans les métadonnées EXIF.
    Score de suspicion 0-100.
    Retourne aussi `detected_software` (str | None).
    """
    try:
        img = Image.open(image_path)
        raw = img._getexif()
    except Exception:
        raw = None

    details = []
    detected_software = None

    if not raw:
        return {
            "score": 50.0,
            "details": ["Aucune métadonnée EXIF — document probablement scanné (neutre)"],
            "detected_software": None,
        }

    exif = {ExifTags.TAGS.get(k, str(k)): str(v) for k, v in raw.items()}
    score = 0.0

    # ── Logiciel de retouche ──
    software_raw = exif.get("Software", "").strip()
    software_lc  = software_raw.lower()
    if software_lc:
        for editor in SOFTWARE_EDITORS:
            if editor in software_lc:
                score += 60
                detected_software = software_raw          # ← nom exact exposé
                details.append(f"Logiciel de retouche détecté : {software_raw}")
                break
        else:
            details.append(f"Logiciel : {software_raw} (non suspect)")

    # ── Incohérence de dates ──
    dt_orig = exif.get("DateTimeOriginal", "").strip()
    dt_digi = exif.get("DateTimeDigitized", "").strip()
    dt_mod  = exif.get("DateTime", "").strip()

    if dt_orig and dt_mod and dt_orig != dt_mod:
        score += 25
        details.append(f"Date modifiée ({dt_mod}) ≠ date originale ({dt_orig})")
    elif dt_orig:
        details.append(f"Dates cohérentes : {dt_orig}")

    if dt_orig and dt_digi and dt_orig != dt_digi:
        score += 10
        details.append("Date de numérisation incohérente avec date originale")

    # ── Tags contradictoires ──
    make  = exif.get("Make", "").strip()
    model = exif.get("Model", "").strip()
    if make and model:
        details.append(f"Appareil : {make} {model}")
    elif software_lc and not make:
        score += 15
        details.append("Logiciel présent sans info appareil photo")

    # ── Orientation modifiée ──
    orientation = exif.get("Orientation", "")
    if orientation and orientation not in ("1", "0"):
        score += 5
        details.append(f"Orientation EXIF modifiée : {orientation}")

    # ── GPS dans un document d'identité ──
    gps = exif.get("GPSInfo", "")
    if gps:
        score += 10
        details.append("Données GPS présentes (inhabituel pour un document d'identité)")

    score = min(score, 100.0)
    if not details:
        details.append("Métadonnées EXIF présentes sans anomalie détectée")

    return {
        "score": float(score),
        "details": details,
        "detected_software": detected_software,
    }


# ─────────────────────────────────────────────
# MODULE 2 — ELA par blocs locaux  +  heatmap
# ─────────────────────────────────────────────

def _analyze_ela(image_path: str) -> dict:
    """
    Error Level Analysis par blocs locaux 16×16.
    Retourne score + heatmap base64 (jet blendée avec l'original).
    """
    try:
        original = _load_rgb(image_path)
        h, w = original.shape[:2]

        buf = io.BytesIO()
        Image.fromarray(original.astype(np.uint8)).save(buf, format="JPEG", quality=75)
        buf.seek(0)
        recompressed = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)

        diff = np.abs(original - recompressed)
        diff_gray = diff.mean(axis=2)

        block_size = 16
        # Construction d'une heatmap pleine résolution par blocs
        heatmap_full = np.zeros((h, w), dtype=np.float32)
        block_errors = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = diff_gray[y:y+block_size, x:x+block_size]
                val = float(block.mean())
                block_errors.append(val)
                heatmap_full[y:y+block_size, x:x+block_size] = val

        if not block_errors:
            return {"score": 0.0, "details": ["ELA : image trop petite"], "ela_base64": None}

        block_errors_arr = np.array(block_errors)
        mean_err = block_errors_arr.mean()
        std_err  = block_errors_arr.std()

        cv = (std_err / mean_err) if mean_err > 1e-6 else 0.0
        threshold = mean_err + 2.0 * std_err
        anomalous_ratio = float((block_errors_arr > threshold).mean())

        score = min(cv * 40 + anomalous_ratio * 100, 100.0)

        ext = os.path.splitext(image_path)[1].lower()
        if ext in (".png",):
            score *= 0.6
        elif ext not in (".jpg", ".jpeg"):
            score *= 0.5

        # ── Génération heatmap ──
        jet_rgb    = _apply_jet_colormap(heatmap_full)
        ela_base64 = _blend_heatmap(original, jet_rgb, alpha=0.60)

        details = [
            f"ELA — erreur moy. {mean_err:.2f}, CV={cv:.3f}, "
            f"blocs anormaux={anomalous_ratio*100:.1f}%"
        ]
        if score >= 40:
            details.append("Zones à erreur élevée détectées — possible retouche locale")
        else:
            details.append("Distribution d'erreur homogène — pas de retouche détectée")

        return {
            "score":      float(min(score, 100.0)),
            "details":    details,
            "ela_base64": ela_base64,
        }

    except Exception as e:
        return {"score": 0.0, "details": [f"ELA échoué : {e}"], "ela_base64": None}


# ─────────────────────────────────────────────
# MODULE 3 — Ghost compression
# ─────────────────────────────────────────────

def _analyze_ghost(image_path: str) -> dict:
    try:
        original = _load_rgb(image_path)
        qualities = [50, 60, 70, 80, 85, 90, 95]
        diffs = []

        for q in qualities:
            buf = io.BytesIO()
            Image.fromarray(original.astype(np.uint8)).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
            diff = float(np.mean(np.abs(original - recomp)))
            diffs.append(diff)

        diffs    = np.array(diffs)
        min_idx  = int(np.argmin(diffs))
        min_q    = qualities[min_idx]
        max_q    = qualities[-1]

        quality_gap = max_q - min_q
        score = min(quality_gap * 1.2, 100.0)

        diff_range = float(diffs.max() - diffs.min())
        if diff_range < 1.0:
            score = 0.0

        details = [
            f"Ghost — qualité minimale à Q={min_q} "
            f"(écart avec Q=95 : {quality_gap})"
        ]
        if score >= 30:
            details.append(
                f"Double compression probable — image déjà sauvegardée à Q≈{min_q}"
            )
        else:
            details.append("Pas de signature de double compression détectée")

        return {"score": float(score), "details": details}

    except Exception as e:
        return {"score": 0.0, "details": [f"Ghost échoué : {e}"]}


# ─────────────────────────────────────────────
# MODULE 4 — Analyse du bruit local  +  heatmap
# ─────────────────────────────────────────────

def _analyze_noise(image_path: str) -> dict:
    """
    Analyse l'uniformité du bruit par blocs 32×32.
    Retourne score + heatmap base64 (jet blendée avec l'original).
    """
    try:
        original = _load_rgb(image_path)
        gray = original.mean(axis=2)
        h, w = gray.shape

        block_size = 32
        heatmap_full = np.zeros((h, w), dtype=np.float32)
        local_stds   = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                val = float(block.std())
                local_stds.append(val)
                heatmap_full[y:y+block_size, x:x+block_size] = val

        if len(local_stds) < 4:
            return {"score": 0.0, "details": ["Bruit : image trop petite"], "noise_base64": None}

        local_stds_arr = np.array(local_stds)
        mean_std = local_stds_arr.mean()
        std_std  = local_stds_arr.std()

        cv = (std_std / mean_std) if mean_std > 1e-6 else 0.0

        low_threshold  = mean_std - 2.0 * std_std
        anomalous_low  = float((local_stds_arr < max(low_threshold, 0.5)).mean())

        high_threshold = mean_std + 2.5 * std_std
        anomalous_high = float((local_stds_arr > high_threshold).mean())

        anomalous_ratio = anomalous_low + anomalous_high
        score = min(cv * 30 + anomalous_ratio * 80, 100.0)

        # ── Génération heatmap ──
        # On inverse : bruit anormalement bas (zones lissées) → rouge
        # On mappe l'écart à la moyenne en valeur absolue
        deviation = np.abs(heatmap_full - mean_std)
        jet_rgb      = _apply_jet_colormap(deviation)
        noise_base64 = _blend_heatmap(original, jet_rgb, alpha=0.55)

        details = [
            f"Bruit — std moy.={mean_std:.2f}, CV={cv:.3f}, "
            f"zones anormales={anomalous_ratio*100:.1f}%"
        ]
        if anomalous_low > 0.05:
            details.append("Zones de bruit anormalement faible détectées (possible copier-coller)")
        if anomalous_high > 0.05:
            details.append("Zones de bruit élevé détectées (possible sur-compression locale)")
        if score < 20:
            details.append("Distribution du bruit homogène — aucune anomalie")

        return {
            "score":       float(score),
            "details":     details,
            "noise_base64": noise_base64,
        }

    except Exception as e:
        return {"score": 0.0, "details": [f"Analyse bruit échouée : {e}"], "noise_base64": None}


# ─────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────

def analyze(image_path: str) -> dict:
    """
    Analyse forensique complète des métadonnées et de l'image.
    Retourne score normalisé 0.0-1.0 + heatmaps ELA & bruit + detected_software.
    """

    W_EXIF  = 0.20
    W_ELA   = 0.35
    W_GHOST = 0.25
    W_NOISE = 0.20

    exif_result  = _analyze_exif(image_path)
    ela_result   = _analyze_ela(image_path)
    ghost_result = _analyze_ghost(image_path)
    noise_result = _analyze_noise(image_path)

    exif_score  = exif_result["score"]
    ela_score   = ela_result["score"]
    ghost_score = ghost_result["score"]
    noise_score = noise_result["score"]

    final_score = (
        exif_score  * W_EXIF  +
        ela_score   * W_ELA   +
        ghost_score * W_GHOST +
        noise_score * W_NOISE
    )
    final_score = float(min(final_score, 100.0))

    if final_score >= 70:
        risk_level = "Critical"
    elif final_score >= 50:
        risk_level = "High"
    elif final_score >= 30:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    diagnostic = []
    for r in [exif_result, ela_result, ghost_result, noise_result]:
        for d in r.get("details", []):
            diagnostic.append(d)

    if final_score >= 50:
        summary = "Anomalies significatives détectées — document potentiellement falsifié."
    elif final_score >= 30:
        summary = "Légères anomalies détectées — vérification recommandée."
    else:
        summary = "Aucune anomalie forensique majeure détectée."

    return {
        "score":                    round(final_score / 100.0, 4),
        "risk_level":               risk_level,
        "summary":                  summary,
        "diagnostic":               diagnostic,
        "ela_score":                round(ela_score, 2),
        "exif_score":               round(exif_score, 2),
        "double_compression_score": round(ghost_score, 2),
        "noise_score":              round(noise_score, 2),
        # ── Nouveaux champs ──
        "detected_software":        exif_result.get("detected_software"),   # str | None
        "ela_base64":               ela_result.get("ela_base64"),           # heatmap ELA
        "noise_base64":             noise_result.get("noise_base64"),       # heatmap bruit
    }