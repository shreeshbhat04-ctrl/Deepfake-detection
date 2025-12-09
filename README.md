# Deepfake Detector

A Hybrid Spatiotemporal System for Misinformation Detection

Status: Deployed & Active
Accuracy: ~85% on Validation Set

This project aims to detect deepfake videos using a hybrid Deep Learning approach, combining Convolutional Neural Networks (CNNs) for spatial feature extraction and Recurrent Neural Networks (RNNs) for temporal analysis. The system is designed to identify manipulation artifacts that are invisible to the naked eye, helping to combat the spread of misinformation.

# Architecture & Evolution

The "Hybrid" Model

Unlike simple image classifiers that look at a single frame, this system analyzes video sequences to detect temporal inconsistencies (jitter, flickering, unnatural blinking).

Spatial Brain (CNN): Uses a pre-trained ResNeXt-50 (Transfer Learning) to extract feature vectors from individual frames.

Temporal Brain (RNN): Uses a Bi-Directional LSTM to analyze the sequence of features over time.

Framework: Core training logic implemented in PyTorch.

# The Evolution of Face Detection


# Phase 1 (OLD SETUP): 
Initially built using Haar Cascade (Haarcascade_frontalface_default.xml). While fast, this method yielded unsatisfactory results due to poor alignment and sensitivity to lighting/angles.

# Phase 2 (Current): 
Swapped to MTCNN (Multi-Task Cascaded Convolutional Networks) using TensorFlow. This upgrade significantly improved accuracy by:

Ignoring background noise.

Providing robust facial landmarks (eyes, nose, mouth) for precise alignment.

Fixing "jitter" in the input data before it reaches the main model.

# Dataset & Performance

Source: Trained on the FaceForensics++ dataset (via Kaggle).

Volume: Processed 800+ video sequences (Balanced Real/Fake split).

Metrics: Achieved an accuracy of 85% on the validation set after 8 epochs of training.

# Deployment Pipeline

This is a fully deployed full-stack application leveraging modern cloud architecture:

Frontend: Dedicated TypeScript/React UI deployed on Vercel.

Backend: Python FastAPI server encapsulated in a Docker Container.

Cloud: Deployed to Google Cloud Platform (GCP) Cloud Run, utilizing serverless autoscaling.

# How to Run It

Option 1: Live Demo

Access the live application here: deepfake-detection-lime.vercel.app

Option 2: Run with Docker (Recommended)

To avoid dependency conflicts, the entire backend is containerized.

Clone the repository:

git clone [https://github.com/your-username/deepfake-detector.git](https://github.com/your-username/deepfake-detector.git)
cd deepfake-detector


Build the Container:

docker build -t deepfake-api .


Run the API:

docker run -p 8080:8080 deepfake-api


# Tech Stack

Core ML: PyTorch, TensorFlow (MTCNN), OpenCV, NumPy, Pandas

Backend: FastAPI, Uvicorn

Frontend: TypeScript, CSS

DevOps: Docker, Google Cloud Run, Vercel, Git LFS


