import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import concurrent.futures

import cnn_module
import ocr_module
import ocr_fields_module
import font_module
import metadata_module


def run_pipeline(image_path: str) -> dict:

    # ── Run all independent modules in parallel ──
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_cnn      = executor.submit(cnn_module.analyze, image_path)
        future_ocr      = executor.submit(ocr_module.analyze, image_path)
        future_font     = executor.submit(font_module.analyze, image_path)
        future_metadata = executor.submit(metadata_module.analyze, image_path)

        cnn_result      = future_cnn.result()
        ocr_result      = future_ocr.result()
        font_result     = future_font.result()
        metadata_result = future_metadata.result()

    # OCR fields depends on ocr_result, run after
    fields_result = ocr_fields_module.analyze(ocr_result)

    # ── MRZ flag ──
    mrz_found = ocr_result.get("mrz_found", True)

    # ── Structural score ──
    mrz_score    = ocr_result["score"]
    fields_score = fields_result["score"]
    font_score   = font_result["score"]

    if mrz_found:
        structural_score = (
            mrz_score    * 0.40 +
            fields_score * 0.40 +
            font_score   * 0.20
        )
    else:
        structural_score = (
            fields_score * 0.55 +
            font_score   * 0.30 +
            mrz_score    * 0.15
        )
        print("ℹ️  No MRZ detected — using visual field weights")

    structural_score = round(max(0.0, min(1.0, structural_score)), 4)

    # ── CNN score ──
    if cnn_result["label"] == "Real":
        cnn_score = cnn_result["confidence"]
    else:
        cnn_score = 1.0 - cnn_result["confidence"]

    # ── Metadata score ──
    metadata_score = 1.0 - metadata_result["score"]

    # ── Global score ──
    if mrz_found:
        global_score = (
            cnn_score        * 0.50 +
            structural_score * 0.30 +
            metadata_score   * 0.20
        )
    else:
        global_score = (
            cnn_score        * 0.55 +
            structural_score * 0.25 +
            metadata_score   * 0.20
        )

    global_score = round(max(0.0, min(1.0, global_score)), 4)

    # ── Verdict ──
    if global_score >= 0.75:
        verdict = "Authentic"
    elif global_score >= 0.50:
        verdict = "Suspicious"
    else:
        verdict = "Fake"

    print(f"✅ Pipeline done — mrz_found={mrz_found}, global_score={global_score} → {verdict}")

    return {
        "global_score":         global_score,
        "global_score_display": round(global_score * 100, 2),
        "verdict":              verdict,
        "structural_score":     structural_score,
        "mrz_found":            mrz_found,
        "cnn":                  cnn_result,
        "ocr":                  ocr_result,
        "ocr_fields":           fields_result,
        "font":                 font_result,
        "metadata":             metadata_result,
    }
