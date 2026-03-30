"""MRZ legacy extraction entrypoint for OCR_check."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Any

# Use LSTM engine (oem 1) - legacy (oem 0) requires extra files
_EXTRA_TESSERACT_PARAMS_LEGACY = "--oem 1"


def read_mrz_legacy(image_path: str, *, save_roi: bool = False) -> Any:
    """Extract MRZ using LSTM Tesseract engine."""
    from ocr_engine.mrz.image import read_mrz

    return read_mrz(
        image_path,
        save_roi=save_roi,
        extra_cmdline_params=_EXTRA_TESSERACT_PARAMS_LEGACY,
    )