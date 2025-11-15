# ~/modeliq/services/api.py
import os
import json
import traceback
import logging
from pathlib import Path
from typing import List, Union
import threading
import subprocess
from contextlib import asynccontextmanager

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from pydantic import BaseModel

# ---------------------------
# Config
# ---------------------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR.parent / "models")).resolve()
MODEL_FILE = MODEL_DIR / os.getenv("MODEL_FILE", "model.pkl")
SCALER_FILE = MODEL_DIR / os.getenv("SCALER_FILE", "scaler.pkl")
META_FILE = MODEL_DIR / os.getenv("META_FILE", "meta.json")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
LOG_FILE = Path(os.getenv("LOG_FILE", BASE_DIR / "api_logs.log"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 5 * 1024 * 1024))  # 5MB

# ensure log dir exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("modeliq_api")

# ---------------------------
# Lifespan (startup/shutdown)
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info(f"🚀 Starting up: loading model artifacts from {MODEL_DIR}...")
        # ensure model dir exists
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

        for p in [MODEL_FILE, SCALER_FILE, META_FILE]:
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        # Load artifacts
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(META_FILE, "r") as fh:
            meta = json.load(fh)

        feature_names = meta.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("meta.json must contain a non-empty 'feature_names' list")

        # Attach to app.state
        app.state.model = model
        app.state.scaler = scaler
        app.state.feature_names = feature_names
        app.state.ready = True

        # Log sklearn version to help diagnose unpickle mismatches
        skl_ver = getattr(sklearn, "__version__", "unknown")
        logger.info(f"✅ Model and metadata loaded. Features: {feature_names}")
        logger.info(f"scikit-learn version in this environment: {skl_ver}")

        yield  # application runs here

    except Exception as exc:
        app.state.ready = False
        logger.error("Failed to load model artifacts during startup:\n" + traceback.format_exc())
        # Re-raise RuntimeError so uvicorn shows the failure and process exits
        raise RuntimeError(f"Failed to load model artifacts: {exc}")

    finally:
        logger.info("🛑 Shutting down API... cleaning up resources.")

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="ModelIQ API",
    description="ModelIQ prediction API",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------
# Middleware: Max Content-Length
# ---------------------------
@app.middleware("http")
async def max_content_length_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_CONTENT_LENGTH:
                detail = f"Payload too large. Max allowed: {MAX_CONTENT_LENGTH} bytes."
                logger.warning(detail)
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"success": False, "error": "payload_too_large", "detail": detail}
                )
        except ValueError:
            # ignore invalid content-length
            pass
    return await call_next(request)

# ---------------------------
# Prediction Request Schema
# ---------------------------
class PredictionRequest(BaseModel):
    data: List[Union[dict, list]]

# ---------------------------
# Exception Handlers
# ---------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "error": "validation_error", "details": exc.errors()}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": "http_error", "details": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: " + traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "internal_server_error", "details": str(exc)}
    )

# ---------------------------
# Health & Readiness
# ---------------------------
@app.get("/health")
def health():
    return {"status": "OK", "app": "ModelIQ API"}

@app.get("/ready")
def ready():
    if getattr(app.state, "ready", False):
        return {"ready": True, "features_count": len(app.state.feature_names)}
    raise HTTPException(status_code=503, detail="Model not loaded")

# ---------------------------
# Helper: assert model ready
# ---------------------------
def assert_model_ready():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Service not ready.")

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        assert_model_ready()
        model = app.state.model
        scaler = app.state.scaler
        feature_names = app.state.feature_names

        if not request.data:
            raise HTTPException(status_code=400, detail="Empty input list.")

        processed = []

        # Handle dict inputs
        if all(isinstance(x, dict) for x in request.data):
            for i, item in enumerate(request.data):
                missing = [f for f in feature_names if f not in item]
                if missing:
                    raise HTTPException(
                        status_code=400,
                        detail={"error": "missing_features", "record": i, "missing": missing}
                    )
                try:
                    row = [float(item[f]) for f in feature_names]
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail={"error": "invalid_numeric_value", "record": i, "data": item}
                    )
                processed.append(row)

        # Handle list inputs
        elif all(isinstance(x, list) for x in request.data):
            for i, row in enumerate(request.data):
                if len(row) != len(feature_names):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "invalid_row_length",
                            "record": i,
                            "expected_length": len(feature_names),
                            "actual_length": len(row)
                        }
                    )
                try:
                    processed.append([float(v) for v in row])
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400,
                        detail={"error": "invalid_numeric_value_in_list", "record": i, "row": row}
                    )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid input format. Must be list of dicts or list of lists."
            )

        X = np.array(processed)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled).tolist()

        resp = {
            "success": True,
            "message": "Prediction completed successfully.",
            "predictions": preds,
            "feature_order": feature_names,
            "n": len(preds)
        }

        if hasattr(model, "predict_proba"):
            try:
                resp["probabilities"] = model.predict_proba(X_scaled).tolist()
            except Exception:
                logger.warning("predict_proba failed; skipping probabilities.")

        logger.info(f"Prediction OK. Count={len(preds)}")
        return resp

    except HTTPException:
        raise
    except Exception:
        logger.error("Prediction exception: " + traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction")

# ---------------------------
# Retrain Endpoint (Background)
# ---------------------------
class RetrainRequest(BaseModel):
    force: bool = False

def run_retrain_script():
    try:
        logger.info("Starting retrain process...")
        # NOTE: retrain_simple_model.py should be accessible from service root
        subprocess.run(["python3", "retrain_simple_model.py"], check=True)
        logger.info("Retrain completed successfully.")
    except Exception:
        logger.error("Retrain failed:\n" + traceback.format_exc())

@app.post("/retrain")
def retrain(request: RetrainRequest):
    threading.Thread(target=run_retrain_script, daemon=True).start()
    return {"success": True, "message": "Retraining started in background."}

# ---------------------------
# Local Dev Entrypoint
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    )
