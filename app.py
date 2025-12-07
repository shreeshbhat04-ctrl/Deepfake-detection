from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import shutil
import tempfile
import torch.nn.functional as F

from model import DeepfakeDetector, FeatureExtractor
from dataset import extract_frames_from_video

SAVED_MODEL_PATH = "saved_models/deepfake_detector_best.pth"
SEQUENCE_LENGTH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

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

model = None
feature_dim = None
model_error: str | None = None


def load_model_if_needed():
    """
    Lazily load the model on first use.
    On Cloud Run this avoids heavy work during container startup.
    """
    global model, feature_dim, model_error

    if model is not None:
        return

    print("⚙️ Loading deepfake model lazily on first request...")
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
            print("❌", err)
            model_error = err
            return

        state = torch.load(SAVED_MODEL_PATH, map_location=DEVICE)
        m.load_state_dict(state)
        m.eval()

        # cache globals only after successful load
        model_error = None
        globals()["model"] = m
        globals()["feature_dim"] = feature_dim_local

        print("✅ Model loaded successfully!")
    except Exception as e:
        model_error = str(e)
        print(f"❌ Error loading model: {e}")


@app.get("/")
def root():
    return {"message": "Deepfake detector backend running"}


@app.get("/health")
def health():
    if model_error is not None:
        return {"status": "model_error", "detail": model_error}
    if model is None:
        return {"status": "model_not_loaded_yet"}
    return {"status": "ok"}


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
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
