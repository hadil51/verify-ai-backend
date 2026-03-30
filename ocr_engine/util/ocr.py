import os
import tempfile

import numpy as np
from imageio import imwrite
import pytesseract

# Set Tesseract path for French Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr(img, mrz_mode=True, extra_cmdline_params=""):
    if img is None or img.shape[-1] == 0:
        return ""

    tmp_dir = tempfile.gettempdir()
    input_file_name = os.path.join(tmp_dir, next(tempfile._get_candidate_names()) + ".bmp")
    output_file_name_base = os.path.join(tmp_dir, next(tempfile._get_candidate_names()))
    output_file_name = output_file_name_base + ".txt"

    try:
        if str(img.dtype).startswith("float") and np.nanmin(img) >= 0 and np.nanmax(img) <= 1:
            img = img.astype(np.float64) * (np.power(2.0, 8) - 1) + 0.499999999
            img = img.astype(np.uint8)

        imwrite(input_file_name, img)

        if mrz_mode:
            config = (
                "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789><"
                " -c load_system_dawg=F -c load_freq_dawg=F {}"
            ).format(extra_cmdline_params)
        else:
            config = "{}".format(extra_cmdline_params)

        pytesseract.pytesseract.run_tesseract(
            input_file_name,
            output_file_name_base,
            "txt",
            lang=None,
            config=config,
        )

        with open(output_file_name, encoding="utf-8") as f:
            return f.read().strip()

    finally:
        for f in [input_file_name, output_file_name]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


def _tempnam():
    tmpfile = tempfile.NamedTemporaryFile(prefix="tess_", delete=False)
    name = tmpfile.name
    tmpfile.close()
    return name