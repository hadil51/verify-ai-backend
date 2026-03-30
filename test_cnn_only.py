"""
Quick CNN test that prints the full error if something goes wrong.
Run from your backend folder:
    python test_cnn_only.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cnn_module

TEST_IMAGE = r"C:\Users\LENOVO\Desktop\PFA_FIN_vers\aze_passport_24_fake_3_13.jpg"

print("Testing cnn_module...")
print(f"Model path : {cnn_module.BEST_MODEL_PATH}")
print(f"Model exists: {os.path.exists(cnn_module.BEST_MODEL_PATH)}")
print(f"Threshold path : {cnn_module.THRESHOLD_PATH}")
print(f"Threshold exists: {os.path.exists(cnn_module.THRESHOLD_PATH)}")
print()

result = cnn_module.analyze(TEST_IMAGE)

print(f"score              : {result['score']}")
print(f"label              : {result['label']}")
print(f"confidence         : {result['confidence']}")
print(f"risk_level         : {result['risk_level']}")
print(f"explanation        : {result['explanation']}")
print(f"threshold_used     : {result['threshold_used']}")
print(f"processing_time_sec: {result['processing_time_sec']}")
print(f"gradcam_base64     : {'✅ present (' + str(len(result['gradcam_base64'])) + ' chars)' if result['gradcam_base64'] else '❌ missing'}")
print(f"original_base64    : {'✅ present (' + str(len(result['original_base64'])) + ' chars)' if result['original_base64'] else '❌ missing'}")

if result['error']:
    print()
    print("=" * 60)
    print("ERROR DETAIL:")
    print(result['error'])
    print("=" * 60)
else:
    print()
    print("✅ CNN module working correctly!")
