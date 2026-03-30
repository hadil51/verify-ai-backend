import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from ocr_engine.mrz_legacy import read_mrz_legacy
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    OCR_IMPORT_ERROR = str(e)


# ─────────────────────────────────────────────
# VISUAL FIELD EXTRACTOR (fallback — no MRZ)
# ─────────────────────────────────────────────

def _extract_fields_visual(image_path: str) -> dict:
    """
    Fallback when no MRZ is detected (e.g. CIN without MRZ).
    Extracts fields via visual OCR using Tesseract.
    """
    import re

    try:
        import pytesseract
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGB")

        h, w = np.array(img).shape[:2]
        if w < 800:
            scale = 800 / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        config = "--oem 1 --psm 6 -l eng+fra"
        raw_text = pytesseract.image_to_string(img, config=config)

    except Exception as e:
        return {"error": f"Visual OCR failed: {e}", "raw_text": ""}

    fields = {"raw_text": raw_text}
    lines  = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    # ── Dates ──
    date_pattern = re.compile(
        r"\b(\d{2}[\/\.\-]\d{2}[\/\.\-]\d{4}|\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2}|\d{8})\b"
    )
    dates_found = []
    for ln in lines:
        for m in date_pattern.finditer(ln):
            dates_found.append(m.group())

    if len(dates_found) >= 1:
        fields["date_of_birth"] = dates_found[0]
    if len(dates_found) >= 2:
        fields["expiration_date"] = dates_found[1]

    # ── Document number ──
    doc_num_pattern = re.compile(r"\b([A-Z0-9]{2}[A-Z0-9\-]{4,12})\b")
    for ln in lines:
        m = doc_num_pattern.search(ln)
        if m and not fields.get("number"):
            fields["number"] = m.group()

    # ── Sex ──
    sex_pattern = re.compile(
        r"\b(M|F|Masculin|Féminin|Male|Female)\b", re.IGNORECASE
    )
    for ln in lines:
        m = sex_pattern.search(ln)
        if m:
            raw_sex = m.group().upper()
            if raw_sex in ("MASCULIN", "MALE", "M"):
                fields["sex"] = "M"
            elif raw_sex in ("FÉMININ", "FEMININ", "FEMALE", "F"):
                fields["sex"] = "F"
            break

    # ── Nationality ──
    nationality_kw = re.compile(
        r"(Nationalit[eé]|Citizenship|Pays)[:\s]+([A-Z]{2,3})", re.IGNORECASE
    )
    for ln in lines:
        m = nationality_kw.search(ln)
        if m:
            fields["nationality"] = m.group(2).upper()
            break

    # ── Latin name detection (uppercase lines) ──
    SKIP_KEYWORDS = {
        "REPUBLIQUE", "REPUBLIC", "NATIONALE", "NATIONAL", "IDENTITE",
        "IDENTITY", "CARD", "CARTE", "PERMIS", "PASSEPORT", "PASSPORT",
        "NOM", "NAME", "PRENOM", "SURNAME", "FIRSTNAME", "DATE",
        "NAISSANCE", "BIRTH", "EXPIRATION", "VALIDITE", "VALIDITY",
        "SEXE", "SEX", "TUNISIE", "TUNISIA", "MAROC", "MOROCCO",
        "ALGERIE", "ALGERIA", "FRANCE", "BELGIQUE", "BELGIUM",
    }
    name_candidates = []
    for ln in lines:
        clean = ln.replace(".", "").replace(",", "").replace("-", " ").strip()
        words = clean.split()
        if (
            all(w.upper() == w and w.isalpha() for w in words)
            and 1 <= len(words) <= 4
            and 2 <= len(clean) <= 40
            and not any(kw in words for kw in SKIP_KEYWORDS)
        ):
            name_candidates.append(clean)

    if len(name_candidates) >= 1:
        fields["surname"] = name_candidates[0]
    if len(name_candidates) >= 2:
        fields["names"] = name_candidates[1]

    return fields


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def analyze(image_path: str) -> dict:
    """
    MRZ OCR analysis.
    If MRZ found  → full MRZ pipeline.
    If not found  → visual OCR fallback.
    """

    # ── Attempt MRZ ──
    if OCR_AVAILABLE:
        try:
            result = read_mrz_legacy(image_path, save_roi=False)
        except Exception:
            result = None
    else:
        result = None

    # ── MRZ found ──
    if result is not None:
        try:
            data = result.to_dict()
        except Exception:
            data = {}

        check_keys = [
            "valid_number",
            "valid_date_of_birth",
            "valid_expiration_date",
            "valid_composite",
            "valid_personal_number",
        ]
        checks = {key: bool(data.get(key, False)) for key in check_keys}
        passed = sum(1 for v in checks.values() if v)
        score  = passed / 5.0

        mrz_type = str(data.get("mrz_type", "")).strip() or "Unknown"

        def safe(key):
            val = data.get(key, "")
            return str(val).strip() if val is not None else ""

        fields = {
            "raw_text":        safe("raw_text"),
            "type":            safe("type"),
            "country":         safe("country"),
            "number":          safe("number"),
            "date_of_birth":   safe("date_of_birth"),
            "expiration_date": safe("expiration_date"),
            "nationality":     safe("nationality"),
            "sex":             safe("sex"),
            "names":           safe("names"),
            "surname":         safe("surname"),
            "personal_number": safe("personal_number"),
            "check_number":    safe("check_number"),
        }

        return {
            "score":       round(score, 4),
            "valid":       all(checks.values()),
            "mrz_found":   True,
            "mrz_type":    mrz_type,
            "valid_score": int(score * 100),
            "fields":      fields,
            "checks":      checks,
            "error":       None,
        }

    # ── No MRZ — visual OCR fallback ──
    visual_fields = _extract_fields_visual(image_path)
    ocr_error = visual_fields.pop("error", None)
    raw_text  = visual_fields.get("raw_text", "")

    FIELD_WEIGHTS = {
        "surname":         0.25,
        "date_of_birth":   0.25,
        "number":          0.20,
        "expiration_date": 0.15,
        "sex":             0.10,
        "nationality":     0.05,
    }
    confidence = sum(
        w for field, w in FIELD_WEIGHTS.items() if visual_fields.get(field)
    )
    score  = round(0.40 + confidence * 0.25, 4)
    checks = {f"field_{k}_found": bool(visual_fields.get(k))
              for k in FIELD_WEIGHTS}

    return {
        "score":       score,
        "valid":       False,
        "mrz_found":   False,
        "mrz_type":    "None",
        "valid_score": int(score * 100),
        "fields":      {**visual_fields, "raw_text": raw_text},
        "checks":      checks,
        "error":       ocr_error,
        "note":        "No MRZ detected — visual field extraction used",
    }