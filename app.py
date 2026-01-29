from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import torch
import os
import shutil
import tempfile
import torch.nn.functional as F
from pathlib import Path

from model import DeepfakeDetector, FeatureExtractor
from dataset import extract_frames_from_video, process_image
from slop_detector import SlopDetector, detect_ai_text, analyze_text_content

BASE_DIR = Path(__file__).resolve().parent
SEQUENCE_LENGTH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Load Models Eagerly ---
    print("Startup: Pre-loading default models to avoid delay...")
    try:
        # Load Video Model
        load_model_if_needed()
        
        # Load Text Model
        load_slop_detector_if_needed()
        print("Startup: All models loaded and ready!")
    except Exception as e:
        print(f"Startup Warning: Could not pre-load models: {e}")
    
    yield
    
    # --- Shutdown (Cleanup if needed) ---
    print("Shutdown: Cleaning up...")

app = FastAPI(lifespan=lifespan)

allowed_origins = [
    "http://localhost:5173",                 # local vite
    "http://localhost:8080",                 # if you're using that
    "https://deepfake-detection-lime.vercel.app/",    # ← replace with real URL after first deploy
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Paths ---
SAVED_MODEL_PATH = BASE_DIR / "saved_models" / "deepfake_detector_best.pth"

model = None
feature_dim = None
model_error: str | None = None

# Slop detector for AI text detection
slop_detector = None
slop_detector_error: str | None = None


# Pydantic models for request/response
class TextAnalysisRequest(BaseModel):
    text: str


class TextAnalysisResponse(BaseModel):
    status: str
    label: str
    confidence: float
    is_ai_generated: bool
    details: Optional[dict] = None


def load_model_if_needed():
    global model, feature_dim, model_error

    if model is not None:
        return

    print("Loading deepfake model lazily on first request...")
    try:
        temp_cnn = FeatureExtractor(freeze=True)
        feature_dim_local = temp_cnn.feature_dim
        del temp_cnn

        m = DeepfakeDetector(
            cnn_feature_dim=feature_dim_local,
            lstm_hidden_size=512,
            lstm_layers=2,
        ).to(DEVICE)

        if not os.path.exists(SAVED_MODEL_PATH):
            err = f"Model file not found at: {SAVED_MODEL_PATH}"
            print("Error:", err)
            model_error = err
            return

        state = torch.load(SAVED_MODEL_PATH, map_location=DEVICE)
        m.load_state_dict(state)
        m.eval()

        # Update globals
        model_error = None
        globals()["model"] = m
        globals()["feature_dim"] = feature_dim_local

        print("Model loaded successfully!")
    except Exception as e:
        model_error = str(e)
        print(f"Error loading model: {e}")


def load_slop_detector_if_needed():
    global slop_detector, slop_detector_error

    if slop_detector is not None:
        return

    print("Loading slop detector for AI text detection...")
    try:
        detector = SlopDetector(device=str(DEVICE))
        detector.load_model()
        
        slop_detector_error = None
        globals()["slop_detector"] = detector
        
        print("Slop detector loaded successfully!")
    except Exception as e:
        slop_detector_error = str(e)
        print(f"Error loading slop detector: {e}")


@app.get("/")
def root():
    return {"message": "Deepfake detector backend running"}


@app.get("/health")
def health():
    status_info = {}
    
    # Check deepfake model status
    if model_error is not None:
        status_info["deepfake_model"] = {"status": "error", "detail": model_error}
    elif model is None:
        status_info["deepfake_model"] = {"status": "not_loaded_yet"}
    else:
        status_info["deepfake_model"] = {"status": "ok"}
    
    # Check slop detector status
    if slop_detector_error is not None:
        status_info["slop_detector"] = {"status": "error", "detail": slop_detector_error}
    elif slop_detector is None:
        status_info["slop_detector"] = {"status": "not_loaded_yet"}
    else:
        status_info["slop_detector"] = {"status": "ok"}
    
    overall_status = "ok"
    if model_error or slop_detector_error:
        overall_status = "partial_error"
    elif model is None and slop_detector is None:
        overall_status = "models_not_loaded_yet"
    
    return {"status": overall_status, "models": status_info}


@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    # Lazy load model on first request
    load_model_if_needed()

    if model is None:
        # loading failed
        raise HTTPException(
            status_code=503,
            detail=f"Model not available on server. Error: {model_error}",
        )

    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload .mp4, .mov, or .avi",
        )

    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        frames_tensor = extract_frames_from_video(
            video_path=temp_file_path,
            sequence_length=SEQUENCE_LENGTH,
        )

        if frames_tensor is None:
            return {
                "status": "error",
                "message": "Could not detect a face in the video.",
            }

        frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(frames_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            prediction_idx = predicted_class.item()
            conf_score = confidence.item() * 100
            result_label = "FAKE" if prediction_idx == 1 else "REAL"

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": result_label,
            "confidence": round(conf_score, 2),
            "is_fake": prediction_idx == 1,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
            os.remove(temp_file_path)


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    # Lazy load model on first request
    load_model_if_needed()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available on server. Error: {model_error}",
        )

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload .jpg, .jpeg, .png, or .webp",
        )

    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # Use the new process_image function
        # This will return a tensor of shape [SEQUENCE_LENGTH, 3, 224, 224]
        # essentially treating the image as a static video
        frames_tensor = process_image(
            image_path=temp_file_path,
            sequence_length=SEQUENCE_LENGTH,
        )

        if frames_tensor is None:
            return {
                "status": "error",
                "message": "Could not detect a face in the image.",
            }

        frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(frames_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            prediction_idx = predicted_class.item()
            conf_score = confidence.item() * 100
            result_label = "FAKE" if prediction_idx == 1 else "REAL"

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": result_label,
            "confidence": round(conf_score, 2),
            "is_fake": prediction_idx == 1,
            "type": "image_analysis"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/analyze-text")
async def analyze_text(request: TextAnalysisRequest):
    load_slop_detector_if_needed()
    
    if slop_detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Slop detector not available. Error: {slop_detector_error}",
        )
    
    try:
        result = slop_detector.detect(request.text)
        
        return {
            "status": "success",
            "label": result.label,
            "confidence": round(result.confidence, 2),
            "is_ai_generated": result.is_ai_generated,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-text-detailed")
async def analyze_text_detailed(request: TextAnalysisRequest):
    load_slop_detector_if_needed()
    
    if slop_detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Slop detector not available. Error: {slop_detector_error}",
        )
    
    try:
        analysis = slop_detector.analyze_paragraphs(request.text)
        
        return {
            "status": "success",
            **analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-combined")
async def predict_combined(
    file: UploadFile = File(...),
    context_text: Optional[str] = Form(None),
):
    # Load both models
    load_model_if_needed()
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Deepfake model not available. Error: {model_error}",
        )

    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload .mp4, .mov, or .avi",
        )

    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # --- Video Deepfake Detection ---
        frames_tensor = extract_frames_from_video(
            video_path=temp_file_path,
            sequence_length=SEQUENCE_LENGTH,
        )

        if frames_tensor is None:
            video_result = {
                "status": "error",
                "message": "Could not detect a face in the video.",
                "prediction": None,
                "confidence": None,
                "is_fake": None,
            }
        else:
            frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(frames_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

                prediction_idx = predicted_class.item()
                conf_score = confidence.item() * 100
                result_label = "FAKE" if prediction_idx == 1 else "REAL"

            video_result = {
                "status": "success",
                "prediction": result_label,
                "confidence": round(conf_score, 2),
                "is_fake": prediction_idx == 1,
            }

        # --- Text Context Analysis (if provided) ---
        text_result = None
        if context_text and context_text.strip():
            load_slop_detector_if_needed()
            
            if slop_detector is not None:
                text_analysis = slop_detector.analyze_paragraphs(context_text)
                text_result = {
                    "status": "success",
                    "overall_label": text_analysis["overall_label"],
                    "overall_confidence": text_analysis["overall_confidence"],
                    "ai_probability": text_analysis["ai_probability"],
                    "paragraph_count": text_analysis["paragraph_count"],
                    "ai_paragraph_count": text_analysis["ai_paragraph_count"],
                }
            else:
                text_result = {
                    "status": "error",
                    "message": f"Slop detector not available: {slop_detector_error}"
                }

        # --- Combined Assessment ---
        combined_verdict = determine_combined_verdict(video_result, text_result)

        return {
            "status": "success",
            "filename": file.filename,
            "video_analysis": video_result,
            "text_analysis": text_result,
            "combined_verdict": combined_verdict,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def determine_combined_verdict(video_result: dict, text_result: Optional[dict]) -> dict:
    video_fake = video_result.get("is_fake")
    video_confidence = video_result.get("confidence", 0)
    video_status = video_result.get("status")
    
    text_ai = None
    text_confidence = None
    
    if text_result and text_result.get("status") == "success":
        text_ai = text_result.get("overall_label") == "AI"
        text_confidence = text_result.get("overall_confidence", 0)
    
    # Determine verdict
    if video_status == "error":
        return {
            "verdict": "INCONCLUSIVE",
            "severity": "unknown",
            "explanation": "Could not analyze video (no face detected). " + 
                          (f"Text appears {'AI-generated' if text_ai else 'human-written'}." if text_ai is not None else "")
        }
    
    if video_fake and text_ai:
        return {
            "verdict": "HIGH_RISK_DEEPFAKE",
            "severity": "high",
            "explanation": f"Video detected as FAKE ({video_confidence:.1f}% confidence) AND associated text appears AI-generated ({text_confidence:.1f}% confidence). This combination suggests sophisticated manipulation."
        }
    elif video_fake and text_ai is False:
        return {
            "verdict": "DEEPFAKE_DETECTED",
            "severity": "high",
            "explanation": f"Video detected as FAKE ({video_confidence:.1f}% confidence). Associated text appears human-written."
        }
    elif video_fake and text_ai is None:
        return {
            "verdict": "DEEPFAKE_DETECTED",
            "severity": "high",
            "explanation": f"Video detected as FAKE ({video_confidence:.1f}% confidence). No text context provided for additional analysis."
        }
    elif not video_fake and text_ai:
        return {
            "verdict": "SUSPICIOUS_CONTEXT",
            "severity": "medium",
            "explanation": f"Video appears REAL ({video_confidence:.1f}% confidence), but associated text appears AI-generated ({text_confidence:.1f}% confidence). Context may be misleading."
        }
    else:
        return {
            "verdict": "LIKELY_AUTHENTIC",
            "severity": "low",
            "explanation": f"Video appears REAL ({video_confidence:.1f}% confidence)." +
                          (f" Associated text appears human-written ({text_confidence:.1f}% confidence)." if text_ai is False else "")
        }
