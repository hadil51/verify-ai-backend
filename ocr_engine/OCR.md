# OCR_check (MRZ legacy only)

## Goal
Provide a dedicated MRZ engine for your app, without relying on `passporteye` imports at runtime.

## What is included
Only the parts needed for MRZ extraction + parsing:

- `OCR_check/mrz/image.py`  
  Contains the full MRZ pipeline classes:
  - `Loader`
  - `Scaler`
  - `BooneTransform`
  - `MRZBoxLocator`
  - `ExtractAllBoxes`
  - `FindFirstValidMRZ`
  - `BoxToMRZ`
  - `TryOtherMaxWidth`
  - `MRZPipeline`
  - `read_mrz(...)`

- `OCR_check/mrz/text.py`  
  MRZ parsing and validation (`MRZ`, OCR cleaner, check-digit logic).

- `OCR_check/util/geometry.py`  
  `RotatedBox` math and ROI extraction.

- `OCR_check/util/pipeline.py`  
  Minimal pipeline execution engine.

- `OCR_check/util/ocr.py`  
  Tesseract OCR bridge (used in legacy mode).

- `OCR_check/mrz_legacy.py`  
  Single app-facing entrypoint:
  - `read_mrz_legacy(image_path, save_roi=False)`  
  which calls `OCR_check.mrz.image.read_mrz(..., extra_cmdline_params="--oem 0")`.

## What was removed as unnecessary
- No CLI scripts.
- No evaluation helpers.
- No PDF extraction path in the app pipeline.
- No runtime import of `passporteye` package name.

## Integration
`website/backend/services/ocr_service.py` calls `read_mrz_legacy`, then converts MRZ output to the same dict format as before (including `valid_score` for global score).

