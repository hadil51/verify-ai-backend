import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# ── Test metadata ──────────────────────────────
print("=" * 40)
print("Testing metadata_module...")
try:
    import metadata_module
    TEST_IMAGE = r"C:\Users\LENOVO\Desktop\PFA_FIN_vers\aze_passport_24_fake_3_13.jpg"
    result = metadata_module.analyze(TEST_IMAGE)
    print("✅ metadata OK")
    print(f"   score                   : {result['score']}")
    print(f"   risk_level              : {result['risk_level']}")
    print(f"   summary                 : {result['summary']}")
    print(f"   ela_score               : {result['ela_score']}")
    print(f"   exif_score              : {result['exif_score']}")
    print(f"   double_compression_score: {result['double_compression_score']}")
    print(f"   diagnostic              :")
    for d in result['diagnostic']:
        print(f"      - {d}")
except Exception as e:
    print(f"❌ metadata FAILED: {e}")

# ── Test OCR ───────────────────────────────────
print("=" * 40)
print("Testing ocr_module...")
try:
    import ocr_module
    TEST_IMAGE = r"C:\Users\LENOVO\Desktop\PFA_FIN_vers\aze_passport_24_fake_3_13.jpg"
    result = ocr_module.analyze(TEST_IMAGE)
    print("✅ ocr OK")
    print(f"   score      : {result['score']}")
    print(f"   valid      : {result['valid']}")
    print(f"   mrz_type   : {result['mrz_type']}")
    print(f"   valid_score: {result['valid_score']}")
    print(f"   error      : {result['error']}")
    print()
    print("   ── CHECKS ──")
    for k, v in result['checks'].items():
        status = "✅" if v else "❌"
        print(f"   {status} {k}: {v}")
    print()
    print("   ── FIELDS ──")
    for k, v in result['fields'].items():
        if v:
            print(f"   {k:20s}: {v}")
except Exception as e:
    print(f"❌ ocr FAILED: {e}")

print("=" * 40)

# ── Test CNN ───────────────────────────────────
print("=" * 40)
print("Testing cnn_module...")
try:
    import cnn_module
    TEST_IMAGE = r"C:\Users\LENOVO\Desktop\PFA_FIN_vers\aze_passport_24_fake_3_13.jpg"
    result = cnn_module.analyze(TEST_IMAGE)
    print("✅ cnn OK")
    print(f"   score      : {result['score']}")
    print(f"   label      : {result['label']}")
    print(f"   confidence : {result['confidence']}")
    print(f"   risk_level : {result['risk_level']}")
    print(f"   explanation: {result['explanation']}")
    print(f"   gradcam    : {'✅ present' if result['gradcam_base64'] else '❌ missing'}")
    print(f"   original   : {'✅ present' if result['original_base64'] else '❌ missing'}")
except Exception as e:
    print(f"❌ cnn FAILED: {e}")

    # ── Test Pipeline ──────────────────────────────
print("=" * 40)
print("Testing pipeline...")
try:
    import pipeline
    TEST_IMAGE = r"C:\Users\LENOVO\Desktop\PFA_FIN_vers\TN_pass.PNG"
    result = pipeline.run_pipeline(TEST_IMAGE)
    print(f"   global_score        : {result['global_score']}")
    print(f"   global_score_display: {result['global_score_display']}")
    print(f"   verdict             : {result['verdict']}")
except Exception as e:
    print(f"❌ pipeline FAILED: {e}")
print("Done.")