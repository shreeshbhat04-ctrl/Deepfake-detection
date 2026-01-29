# Deepfake Detection with AI Text Analysis

This project aims to detect deepfakes using neural networks such as CNN's and RNN's, combined with AI-generated text detection for multi-modal analysis.

## Features

### 🎥 Video Deepfake Detection
- **ResNeXt50 + Bidirectional LSTM** architecture for temporal analysis
- Frame-level face detection using MTCNN
- Real-time inference via FastAPI backend

### 📝 AI Text Detection (NEW)
- Integrated **slop-detector-bert** model from Hugging Face
- Detects AI-generated text in transcripts, captions, or metadata
- Paragraph-level analysis with aggregated confidence scores

### 🔄 Combined Analysis
- Multi-modal deepfake detection (video + text)
- Risk-level assessment combining both analyses
- Detailed breakdown of individual component results

## API Endpoints

### Video Analysis
```bash
POST /predict
# Upload video file for deepfake detection
```

### Text Analysis
```bash
POST /analyze-text
# Analyze text for AI-generated content
Body: {"text": "Your text here"}
```

### Combined Analysis (Recommended)
```bash
POST /predict-combined
# Combined video deepfake + text AI detection
Form Data:
  - file: video file (.mp4, .mov, .avi)
  - context_text: (optional) transcript or caption text
```

## Installation

### Backend
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app:app --reload
```

### Frontend
```bash
cd frontend/deepfake-reveal
npm install
npm run dev
```

## Architecture

### Deepfake Model
- **Feature Extractor**: ResNeXt50 pre-trained on ImageNet
- **Sequence Model**: 2-layer Bidirectional LSTM (512 hidden units)
- **Classification Head**: Fully connected layers with dropout

### Text Analysis Model
- **Model**: `gouwsxander/slop-detector-bert` (BERT-base with LoRA)
- **Task**: Binary classification (HUMAN vs AI)
- **Accuracy**: 96% on Wikipedia-style text

## Combined Verdict Logic

| Video Result | Text Result | Combined Verdict |
|-------------|-------------|------------------|
| FAKE | AI-generated | HIGH_RISK_DEEPFAKE |
| FAKE | Human-written | DEEPFAKE_DETECTED |
| REAL | AI-generated | SUSPICIOUS_CONTEXT |
| REAL | Human-written | LIKELY_AUTHENTIC |

## Credits

- Deepfake detection based on CNN-LSTM architecture
- AI text detection powered by [slop-detector-bert](https://huggingface.co/gouwsxander/slop-detector-bert)
- Reference: [gouwsxander/slop-detector](https://github.com/gouwsxander/slop-detector)
