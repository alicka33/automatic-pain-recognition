# Pain Detection Server

Real-time pain classification API based on facial video analysis using MediaPipe and LSTM models.

## Overview

This server provides a REST API that:
1. Accepts video uploads via HTTP POST
2. Processes videos using MediaPipe to extract facial landmarks
3. Optionally Computes Euclidean distances from a reference point
4. Runs inference using a trained LSTM model
5. Returns pain level classification with confidence scores

## Architecture

```
Video Upload
    ↓
MediaPipe Landmark Extraction
    ↓
Frontalization (optional, Procrustes alignment)
    ↓
Feature Engineering (Euclidean distances)
    ↓
Feature Selection (top-N landmarks)
    ↓
Sequence Padding/Truncation
    ↓
LSTM Model Inference
    ↓
Pain Classification Result
```

## Setup

### 1. Install Dependencies

```bash
cd pain_detection_app_server
pip install -r requirements.txt
```

Required packages:
- fastapi
- uvicorn
- torch
- numpy
- opencv-python
- mediapipe
- pandas

### 2. Configure the Server

Edit `config.py` to match your trained model:

```python
# Model checkpoint path
MODEL_CHECKPOINT_PATH = MODEL_DIR / "best_pain_model.pt"

# Model architecture (must match training)
NUM_FEATURES = 100  # or 1434, or 300 or depending on your setup
NUM_CLASSES = 2     # 2 for binary, 5 for multiclass
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# Choose model type
MODEL_TYPE = "bi_lstm"  # or "attention_lstm" or "sta_lstm"
```

### 3. Add Model Files

Place your trained checkpoint in `model/model_paths/` and set `MODEL_CHECKPOINT_PATH` in `config.py` (defaults to `testing_new_code_sta_lstm_2_classes_300_coord.pt`). Assets included:
- Model checkpoint examples: `model/model_paths/testing_new_code_sta_lstm_*.pt`
- Reference keypoints: `data/frontalization/key_points_xyz.npy`
- Feature indices: `data/landmarks/top_100_important_landmarks_emotions.npy`

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Pain Detection Server is running",
  "status": "ready",
  "model_type": "bi_lstm",
  "num_classes": 2
}
```

### `GET /health`
Detailed server status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "config": {
    "model_type": "bi_lstm",
    "num_classes": 2,
    "num_features": 100,
    "max_sequence_length": 46
  }
}
```

### `POST /upload-video`
Main endpoint for pain classification.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: video file (mp4, avi, mov, etc.)

**Response:**
```json
{
  "painLevel": "NO_PAIN",
  "confidence": 0.87,
  "numFrames": 23,
  "probabilities": {
    "NO_PAIN": 0.13,
    "VERY_STRONG_PAIN": 0.87
  },
  "description": "No Pain Detected. No signs of discomfort observed."
}
```

## File Structure

```
pain_detection_app_server/
├── app.py                          # FastAPI application
├── config.py                       # Configuration file
├── Dockerfile
├── requirements.txt
├── data/
│   ├── frontalization/
│   │   └── key_points_xyz.npy
│   └── landmarks/
│       └── top_100_important_landmarks_emotions.npy
├── model/
│   ├── Attention_LSTM.py
│   ├── Bi_LSTM.py
│   ├── STA_LSTM.py
│   └── model_paths/                # Stored checkpoints
├── services/
│   ├── file_handler.py
│   ├── model_loader.py
│   ├── pain_detection_service.py
│   ├── processing_pipeline_mediapipe.py
│   └── video_inference.py
├── tests/
│   └── test_setup_colab.ipynb
└── README.md                       # This file
```

## License

Part of the Automatic Pain Recognition project.
