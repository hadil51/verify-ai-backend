import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

try:
    from skimage import filters, morphology, measure
    from skimage import io as skimage_io
    from skimage.color import rgb2gray
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


def _load_gray(image_path: str) -> np.ndarray | None:
    try:
        # Méthode 1 : imageio (plus robuste)
        import imageio
        img = imageio.imread(image_path)
        if img is None:
            raise ValueError("imageio returned None")
        # Gère RGBA, RGB, L
        if img.ndim == 3 and img.shape[2] == 4:
            # RGBA → retire le canal alpha puis convertit
            img = img[:, :, :3]
        if img.ndim == 3:
            img = rgb2gray(img)
        return img.astype(np.float64)
    except Exception as e1:
        try:
            # Méthode 2 : PIL comme fallback
            from PIL import Image
            img = Image.open(image_path).convert("L")
            return np.array(img, dtype=np.float64) / 255.0
        except Exception as e2:
            print(f"⚠️ font_module: impossible de charger {image_path}: {e1} | {e2}")
            return None


def _extract_mrz_region(img: np.ndarray) -> np.ndarray:
    """Prend les 22% inférieurs de l'image (zone MRZ standard)."""
    h = img.shape[0]
    start = int(h * 0.78)
    return img[start:, :]


def _binarize(img: np.ndarray) -> np.ndarray:
    thresh = filters.threshold_otsu(img)
    return img < thresh  # texte sombre sur fond clair → True = texte


def _get_char_components(binary: np.ndarray):
    """Retourne les propriétés des composantes connexes (caractères)."""
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    # Filtre les composantes trop petites ou trop grandes
    h, w = binary.shape
    min_area = max(4, h * w * 0.0002)
    max_area = h * w * 0.15
    filtered = [p for p in props if min_area < p.area < max_area]
    return filtered


def analyze(image_path: str) -> dict:
    """
    Analyse la cohérence des polices et l'alignement de la zone MRZ.
    Retourne un score 0.0-1.0 + détail des checks.
    """

    if not SKIMAGE_OK:
        return {
            "score": 0.5,
            "checks": {},
            "details": {"error": "scikit-image non disponible"},
            "error": "scikit-image non disponible"
        }

    img = _load_gray(image_path)
    if img is None:
        return {
            "score": 0.5,          # ← 0.5 au lieu de 0.0 (neutre, pas pénalisé)
            "checks": {},
            "details": {"info": "Analyse police non disponible pour ce format"},
            "error": None          # ← None pour ne pas afficher d'erreur rouge
        }

    mrz = _extract_mrz_region(img)
    if mrz.size == 0:
        return {
            "score": 0.0,
            "checks": {},
            "details": {},
            "error": "Zone MRZ introuvable"
        }

    try:
        binary = _binarize(mrz)
    except Exception:
        return {
            "score": 0.5,
            "checks": {},
            "details": {"error": "Binarisation échouée"},
            "error": None
        }

    props = _get_char_components(binary)
    checks = {}
    details = {}

    if len(props) < 5:
        return {
            "score": 0.0,
            "checks": {"enough_components": False},
            "details": {"enough_components": f"Trop peu de composantes détectées : {len(props)}"},
            "error": None
        }

    checks["enough_components"] = True
    details["enough_components"] = f"{len(props)} composantes détectées"

    heights   = np.array([p.bbox[2] - p.bbox[0] for p in props], dtype=float)
    widths    = np.array([p.bbox[3] - p.bbox[1] for p in props], dtype=float)
    centroids = np.array([p.centroid for p in props])

    # ── 1. Uniformité de la hauteur des caractères ──
    h_mean = heights.mean()
    h_cv   = heights.std() / h_mean if h_mean > 0 else 1.0
    # CV < 0.25 = uniforme, CV > 0.5 = suspect
    checks["uniform_height"] = h_cv < 0.35
    details["uniform_height"] = (
        f"Hauteur moy. {h_mean:.1f}px, CV={h_cv:.3f} "
        f"({'OK' if checks['uniform_height'] else 'Irrégulier'})"
    )

    # ── 2. Uniformité de la largeur des caractères ──
    w_mean = widths.mean()
    w_cv   = widths.std() / w_mean if w_mean > 0 else 1.0
    checks["uniform_width"] = w_cv < 0.40
    details["uniform_width"] = (
        f"Largeur moy. {w_mean:.1f}px, CV={w_cv:.3f} "
        f"({'OK' if checks['uniform_width'] else 'Irrégulier'})"
    )

    # ── 3. Alignement horizontal (centroïdes sur même ligne) ──
    y_coords = centroids[:, 0]
    # Clustérise en 1 ou 2 lignes (TD3 = 2 lignes, TD1 = 3)
    # On mesure l'écart-type résiduel après détection des lignes
    if len(y_coords) > 10:
        # Tri et détection de sauts (séparation entre lignes)
        y_sorted = np.sort(y_coords)
        gaps = np.diff(y_sorted)
        median_gap = np.median(gaps)
        large_gaps = gaps[gaps > 5 * median_gap]
        n_lines = len(large_gaps) + 1
        # Calcule la variation autour de la moyenne par ligne
        line_variances = []
        split_points = [0] + list(np.where(gaps > 5 * median_gap)[0] + 1) + [len(y_sorted)]
        for i in range(len(split_points) - 1):
            line_y = y_sorted[split_points[i]:split_points[i+1]]
            if len(line_y) > 1:
                line_variances.append(line_y.std())
        avg_line_var = np.mean(line_variances) if line_variances else 0
        h_img = mrz.shape[0]
        # Normalisation par la hauteur de l'image
        alignment_score_val = avg_line_var / h_img
        checks["horizontal_alignment"] = alignment_score_val < 0.08
        details["horizontal_alignment"] = (
            f"{n_lines} ligne(s) détectée(s), "
            f"variation alignement={alignment_score_val:.4f} "
            f"({'OK' if checks['horizontal_alignment'] else 'Désaligné'})"
        )
    else:
        checks["horizontal_alignment"] = True
        details["horizontal_alignment"] = "Pas assez de points pour mesurer l'alignement"

    # ── 4. Espacement horizontal régulier ──
    x_coords = np.sort(centroids[:, 1])
    if len(x_coords) > 3:
        spacings = np.diff(x_coords)
        # Filtre les grands écarts (entre les deux lignes groupées)
        median_sp  = np.median(spacings)
        normal_sp  = spacings[spacings < 4 * median_sp]
        if len(normal_sp) > 1:
            sp_cv = normal_sp.std() / normal_sp.mean() if normal_sp.mean() > 0 else 1.0
            checks["regular_spacing"] = sp_cv < 0.50
            details["regular_spacing"] = (
                f"Espacement moy. {normal_sp.mean():.1f}px, CV={sp_cv:.3f} "
                f"({'OK' if checks['regular_spacing'] else 'Irrégulier'})"
            )
        else:
            checks["regular_spacing"] = True
            details["regular_spacing"] = "Espacement non mesurable"
    else:
        checks["regular_spacing"] = True
        details["regular_spacing"] = "Pas assez de composantes"

    # ── 5. Densité de pixels uniforme par caractère ──
    densities = np.array([p.area / ((p.bbox[2]-p.bbox[0]) * (p.bbox[3]-p.bbox[1]) + 1e-9) for p in props])
    d_cv = densities.std() / densities.mean() if densities.mean() > 0 else 1.0
    checks["uniform_density"] = d_cv < 0.45
    details["uniform_density"] = (
        f"Densité moy. {densities.mean():.3f}, CV={d_cv:.3f} "
        f"({'OK' if checks['uniform_density'] else 'Densité irrégulière — possible falsification'})"
    )

    # ── Score final ──
    passed = sum(1 for v in checks.values() if v)
    total  = len(checks)
    score  = round(passed / total, 4)

    return {
        "score":   score,
        "checks":  checks,
        "details": details,
        "error":   None
    }