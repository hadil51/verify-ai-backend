import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import tempfile
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import gdown
import pipeline

# ─────────────────────────────────────────────
# AUTO-DOWNLOAD MODEL IF MISSING
# ─────────────────────────────────────────────

MODEL_PATH = "ID_Project/models/best_resnet50_id.h5"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(
       
        id="1Q2tlsv-EIt_RlunFcoHfh0jdYSqx8AIq",
        output=MODEL_PATH,
        quiet=False,
        fuzzy=True
    )
    print("✅ Model downloaded!")

# ─────────────────────────────────────────────
# JSON serializer qui gère numpy
# ─────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def numpy_safe(data):
    """Convertit récursivement tous les types numpy en types Python natifs."""
    return json.loads(json.dumps(data, cls=NumpyEncoder))

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(title="Document Authenticity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "https://verify-ai-vers.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    original_name = file.filename or "upload.jpg"
    ext = os.path.splitext(original_name)[1].lower()
    if not ext:
        ext = ".jpg"

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        print(f"📁 Analyzing: {original_name} ({len(contents)} bytes)")

        result = pipeline.run_pipeline(tmp.name)
        result = numpy_safe(result)

        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        try:
            os.remove(tmp.name)
        except:
            pass