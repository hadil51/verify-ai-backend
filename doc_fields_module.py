"""
doc_fields_module.py
====================
Field extraction using:
  - Simple preprocessing: grayscale → upscale → denoise → adaptive threshold
  - Tesseract TSV mode: bounding-box-aware label → value spatial lookup
  - Person photo: OpenCV face detection + heuristic fallback

Bugs fixed vs previous version:
  1. _value_after_label: condition `next_y - y <= (next_y - y) * 2.5` was
     always True. Fixed to a concrete 60px threshold.
  2. _build_rows: conf column can be "-1" string or NaN. Fixed with safe cast.
  3. Label matching: Tesseract often OCRs uppercase ("NOM" not "nom").
     Fixed by lowercasing + stripping punctuation before comparing.
  4. DATE_RE: \\b fails on "/" and "." separators. Fixed with lookahead.
  5. DOCNUM_RE: was too greedy, matched country codes. Tightened pattern.
  6. _build_rows: ±10px grouping too tight at 1200px. Increased to ±15px.
  7. Multi-word labels ("date de naissance") were never matched. Added
     phrase lookup pass before single-word lookup.
"""

import io
import os
import re
import base64
from typing import Optional

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_pil(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def _preprocess_for_ocr(image_path: str) -> Image.Image:
    """
    4-step preprocessing: grayscale → upscale → denoise → adaptive threshold.
    Falls back to raw PIL upscale if OpenCV unavailable.
    """
    try:
        import cv2

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cv2 imread returned None")

        h, w = img.shape
        if w < 1200:
            img = cv2.resize(
                img,
                (1200, int(h * 1200 / w)),
                interpolation=cv2.INTER_CUBIC,
            )

        img = cv2.fastNlMeansDenoising(img, h=10)

        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=11,
        )

        return Image.fromarray(img)

    except Exception as e:
        print(f"⚠️  Preprocessing failed ({e}), using raw image")
        img = Image.open(image_path).convert("L")
        w, h = img.size
        if w < 1200:
            img = img.resize((1200, int(h * 1200 / w)), Image.LANCZOS)
        return img


# ─────────────────────────────────────────────
# TSV BOUNDING BOX EXTRACTION
# ─────────────────────────────────────────────

LABEL_KEYWORDS: dict[str, str] = {
    "nom":          "surname",
    "name":         "surname",
    "surname":      "surname",
    "last":         "surname",
    "prenom":       "names",
    "prénom":       "names",
    "prénoms":      "names",
    "given":        "names",
    "first":        "names",
    "firstname":    "names",
    "naissance":    "date_of_birth",
    "birth":        "date_of_birth",
    "né":           "date_of_birth",
    "née":          "date_of_birth",
    "dob":          "date_of_birth",
    "expiration":   "expiration_date",
    "expiry":       "expiration_date",
    "validité":     "expiration_date",
    "validity":     "expiration_date",
    "expires":      "expiration_date",
    "numéro":       "number",
    "numero":       "number",
    "number":       "number",
    "n°":           "number",
    "sexe":         "sex",
    "sex":          "sex",
    "genre":        "sex",
    "gender":       "sex",
    "nationalité":  "nationality",
    "nationality":  "nationality",
    "nat":          "nationality",
    "pays":         "nationality",
}

# Multi-word label phrases checked as consecutive words on the same line
LABEL_PHRASES: dict[tuple, str] = {
    ("date", "de", "naissance"):  "date_of_birth",
    ("date", "naissance"):        "date_of_birth",
    ("date", "of", "birth"):      "date_of_birth",
    ("date", "d", "expiration"):  "expiration_date",
    ("date", "expiration"):       "expiration_date",
    ("date", "of", "expiry"):     "expiration_date",
    ("last", "name"):             "surname",
    ("first", "name"):            "names",
    ("given", "name"):            "names",
    ("document", "number"):       "number",
    ("document", "no"):           "number",
    ("doc", "number"):            "number",
}

# FIX 4: no \\b around separators
DATE_RE   = re.compile(
    r"(?<!\d)(\d{2}[\/\.\-]\d{2}[\/\.\-]\d{4}|\d{4}[\/\.\-]\d{2}[\/\.\-]\d{2})(?!\d)"
)
# FIX 5: must contain at least one digit, min 6 chars
DOCNUM_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{0,2}[0-9][A-Z0-9]{5,11})(?![A-Z0-9])")
SEX_RE    = re.compile(r"\b(Masculin|Féminin|Male|Female)\b", re.IGNORECASE)

NAME_SKIP = {
    "REPUBLIQUE", "REPUBLIC", "NATIONALE", "NATIONAL", "IDENTITE", "IDENTITY",
    "CARD", "CARTE", "PERMIS", "PASSEPORT", "PASSPORT", "NOM", "NAME", "PRENOM",
    "SURNAME", "FIRSTNAME", "DATE", "NAISSANCE", "BIRTH", "EXPIRATION", "VALIDITE",
    "VALIDITY", "SEXE", "SEX", "TUNISIE", "TUNISIA", "MAROC", "MOROCCO",
    "ALGERIE", "ALGERIA", "FRANCE", "BELGIQUE", "BELGIUM", "DOCUMENT",
    "NATIONALITY", "NATIONALITE", "GENRE", "GENDER",
}


def _safe_conf(val) -> int:
    """FIX 2: conf can be '-1' string or NaN."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return -1


def _clean_word(text: str) -> str:
    """FIX 3: lowercase + strip trailing punctuation for label matching."""
    return text.lower().strip().rstrip(":./,;")


def _build_rows(tsv_df) -> dict[int, list[dict]]:
    """
    Groups TSV words into lines by top-y coordinate.
    FIX 2: safe conf cast.
    FIX 6: ±15px grouping tolerance (was ±10px, too tight at 1200px).
    """
    rows: dict[int, list[dict]] = {}
    for _, row in tsv_df.iterrows():
        if _safe_conf(row["conf"]) < 20:
            continue
        text = str(row["text"]).strip()
        if not text:
            continue
        top = int(row["top"])
        key = next((k for k in rows if abs(k - top) <= 15), top)  # FIX 6
        rows.setdefault(key, []).append({
            "text":  text,
            "left":  int(row["left"]),
            "top":   top,
            "width": int(row["width"]),
        })
    for key in rows:
        rows[key].sort(key=lambda w: w["left"])
    return rows


def _value_after_label(rows: dict, keyword: str) -> Optional[str]:
    """
    Single-word label lookup.
    FIX 1: next-line condition now uses concrete 60px threshold.
    FIX 3: uses _clean_word() before comparing.
    """
    sorted_rows = sorted(rows.items())
    for row_idx, (y, words) in enumerate(sorted_rows):
        for w_idx, word in enumerate(words):
            if _clean_word(word["text"]) == keyword:
                label_x = word["left"] + word["width"]

                # Same line — words to the right
                right = [w["text"] for w in words[w_idx + 1:]
                         if w["left"] >= label_x - 5]
                if right:
                    return " ".join(right)

                # FIX 1: next line within 60px
                if row_idx + 1 < len(sorted_rows):
                    next_y, next_words = sorted_rows[row_idx + 1]
                    if next_y - y <= 60:
                        below = [w["text"] for w in next_words
                                 if w["left"] >= word["left"] - 20]
                        if below:
                            return " ".join(below)
    return None


def _value_after_phrase(rows: dict, phrase: tuple) -> Optional[str]:
    """
    FIX 7: multi-word label lookup (e.g. "date de naissance").
    Scans each line for consecutive words matching the phrase,
    returns words to the right or on the next line.
    """
    sorted_rows = sorted(rows.items())
    phrase_len  = len(phrase)

    for row_idx, (y, words) in enumerate(sorted_rows):
        cleaned = [_clean_word(w["text"]) for w in words]
        for i in range(len(cleaned) - phrase_len + 1):
            if tuple(cleaned[i:i + phrase_len]) == phrase:
                last_word = words[i + phrase_len - 1]
                label_x   = last_word["left"] + last_word["width"]

                right = [w["text"] for w in words[i + phrase_len:]
                         if w["left"] >= label_x - 5]
                if right:
                    return " ".join(right)

                if row_idx + 1 < len(sorted_rows):
                    next_y, next_words = sorted_rows[row_idx + 1]
                    if next_y - y <= 60:
                        below = [w["text"] for w in next_words
                                 if w["left"] >= words[i]["left"] - 20]
                        if below:
                            return " ".join(below)
    return None


def _extract_fields_tsv(preprocessed_img: Image.Image) -> dict:
    """
    4-pass field extraction:
    Pass 1a — multi-word phrase spatial lookup  (most reliable)
    Pass 1b — single-word label spatial lookup
    Pass 2  — regex fallback for dates / doc number / sex
    Pass 3  — uppercase line heuristic for names only
    """
    try:
        import pytesseract
        tsv = pytesseract.image_to_data(
            preprocessed_img,
            config="--oem 1 --psm 6 -l eng+fra",
            output_type=pytesseract.Output.DATAFRAME,
        )
    except Exception as e:
        return {"error": f"TSV OCR failed: {e}"}

    rows   = _build_rows(tsv)
    fields: dict = {}
    found:  set  = set()

    # ── Pass 1a: multi-word phrase lookup ──
    for phrase, field_name in LABEL_PHRASES.items():
        if field_name in found:
            continue
        value = _value_after_phrase(rows, phrase)
        if value:
            fields[field_name] = value.strip()
            found.add(field_name)

    # ── Pass 1b: single-word label lookup ──
    for keyword, field_name in LABEL_KEYWORDS.items():
        if field_name in found:
            continue
        value = _value_after_label(rows, keyword)
        if value:
            fields[field_name] = value.strip()
            found.add(field_name)

    # ── Pass 2: regex fallback ──
    flat = " ".join(w["text"] for words in rows.values() for w in words)

    if "date_of_birth" not in found or "expiration_date" not in found:
        dates = DATE_RE.findall(flat)
        if dates and "date_of_birth" not in found:
            fields["date_of_birth"] = dates[0]
        if len(dates) >= 2 and "expiration_date" not in found:
            fields["expiration_date"] = dates[1]

    if "number" not in found:
        m = DOCNUM_RE.search(flat)
        if m:
            fields["number"] = m.group()

    if "sex" not in found:
        m = SEX_RE.search(flat)
        if m:
            s = m.group().upper()
            if s in ("MASCULIN", "MALE"):
                fields["sex"] = "M"
            elif s in ("FÉMININ", "FEMININ", "FEMALE"):
                fields["sex"] = "F"
        else:
            # Standalone M/F only when it appears right after a sex label
            sorted_rows = sorted(rows.items())
            for row_idx, (y, words) in enumerate(sorted_rows):
                for w_idx, word in enumerate(words):
                    if _clean_word(word["text"]) in ("sexe", "sex", "genre", "gender"):
                        candidates = words[w_idx + 1:]
                        if not candidates and row_idx + 1 < len(sorted_rows):
                            _, candidates = sorted_rows[row_idx + 1]
                        for c in candidates:
                            if c["text"].upper() in ("M", "F"):
                                fields["sex"] = c["text"].upper()
                                found.add("sex")
                                break
                        break

    # ── Pass 3: uppercase line fallback for names ──
    if "surname" not in found or "names" not in found:
        name_candidates = []
        for y, words in sorted(rows.items()):
            line  = " ".join(w["text"] for w in words)
            clean = line.replace(".", "").replace(",", "").replace("-", " ").split()
            if (
                all(w.upper() == w and w.isalpha() for w in clean)
                and 1 <= len(clean) <= 4
                and 2 <= len(line) <= 40
                and not any(kw in clean for kw in NAME_SKIP)
            ):
                name_candidates.append(line.strip())
        if "surname" not in found and name_candidates:
            fields["surname"] = name_candidates[0]
        if "names" not in found and len(name_candidates) >= 2:
            fields["names"] = name_candidates[1]

    return fields


# ─────────────────────────────────────────────
# PHOTO EXTRACTION
# ─────────────────────────────────────────────

def _extract_photo(image_path: str, img_pil: Image.Image) -> str:
    """OpenCV face detection with heuristic top-left fallback."""
    try:
        import cv2
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError("cv2 imread failed")

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        for cp in [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
        ]:
            if not os.path.exists(cp):
                continue
            faces = cv2.CascadeClassifier(cp).detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
            )
            if len(faces) > 0:
                h, w = img_cv.shape[:2]
                x, y, fw, fh = sorted(
                    faces, key=lambda f: f[2] * f[3], reverse=True
                )[0]
                x1 = max(0, x - int(fw * 0.4))
                y1 = max(0, y - int(fh * 0.5))
                x2 = min(w, x + fw + int(fw * 0.4))
                y2 = min(h, y + fh + int(fh * 0.5))
                cropped = img_pil.crop((x1, y1, x2, y2))
                return _to_base64(cropped.resize((160, 200), Image.LANCZOS))

    except Exception:
        pass

    # Heuristic fallback — photo is top-left on most ID cards
    w, h = img_pil.size
    cropped = img_pil.crop((int(w * 0.02), int(h * 0.08), int(w * 0.32), int(h * 0.62)))
    return _to_base64(cropped.resize((160, 200), Image.LANCZOS))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

FIELD_LABELS: dict[str, str] = {
    "surname":         "Last name",
    "names":           "First name(s)",
    "date_of_birth":   "Date of birth",
    "expiration_date": "Expiry date",
    "number":          "Document №",
    "sex":             "Sex",
    "nationality":     "Nationality",
}


def analyze(image_path: str) -> dict:
    img_pil      = _load_pil(image_path)
    photo_b64    = _extract_photo(image_path, img_pil)
    preprocessed = _preprocess_for_ocr(image_path)
    raw_fields   = _extract_fields_tsv(preprocessed)
    ocr_error    = raw_fields.pop("error", None)

    clean_fields = {
        k: v for k, v in raw_fields.items()
        if v and k in FIELD_LABELS
    }

    return {
        "photo_base64": photo_b64,
        "fields":       clean_fields,
        "field_labels": {k: FIELD_LABELS[k] for k in clean_fields},
        "error":        ocr_error,
    }
