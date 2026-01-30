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

## Deployment on Hugging Face Spaces

This server is designed to run on Hugging Face Spaces for easy GPU access and cloud deployment.

### Prerequisites
- Hugging Face account (free)
- Your trained model checkpoint
- GitHub repository with this code

### Steps to Deploy

1. **Create a new Space on Hugging Face:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Docker" as the runtime
   - Set repository visibility to **Private** (if you want to restrict access)

2. **Add your model files:**
   - Upload your trained model checkpoint to `model/model_paths/`
   - Ensure all data files (keypoints, landmarks) are included in the repo

3. **Configure the server in `config.py`:**
   ```python
   # Model checkpoint path (must match your uploaded file)
   MODEL_CHECKPOINT_PATH = MODEL_DIR / "model_paths/your_model.pt"
   
   # Model architecture (must match training configuration)
   NUM_FEATURES = 300      # e.g., 100, 300, or 1434
   NUM_CLASSES = 2         # 2 for binary, 5 for multiclass
   HIDDEN_SIZE = 128
   NUM_LAYERS = 2
   MODEL_TYPE = "sta_lstm" # "bi_lstm", "attention_lstm", or "sta_lstm"
   ```

4. **Deploy:**
   - Hugging Face will automatically detect the Dockerfile and build the container
   - Once deployed, you'll get a URL: `https://yourusername-yourspacename.hf.space`

### Connect the App to the Server

Configure your React Native app (`pain_detection_app/`) to connect to the deployed server.

#### Option A: Private Space (Requires Authentication)

**Current setup** - Server is private and needs a token:

1. **Set the server URL in `constants/api.ts`:**
   ```typescript
   export const API_UPLOAD_URL = 'https://yourusername-yourspacename.hf.space/upload-video';
   export const HUGGING_FACE_TOKEN = process.env.EXPO_PUBLIC_HUGGING_FACE_TOKEN;
   ```

2. **Get your Hugging Face token:**
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with **read** access

3. **Add token to app environment:**
   - Create a `.env` file in the `pain_detection_app` folder:
     ```
     EXPO_PUBLIC_HUGGING_FACE_TOKEN=hf_your_token_here
     ```

4. **The app automatically sends the token** in `services/videoAnalysis.ts`:
   ```typescript
   const response = await fetch(API_UPLOAD_URL, {
     method: 'POST',
     headers: { Authorization: `Bearer ${HUGGING_FACE_TOKEN}` },
     body: formData,
   });
   ```

#### Option B: Public Space (No Authentication)

If you make your Space public, you can simplify the app:

1. **Update `constants/api.ts`:**
   ```typescript
   export const API_UPLOAD_URL = 'https://yourusername-yourspacename.hf.space/upload-video';
   // No token needed for public spaces
   ```

2. **Update `services/videoAnalysis.ts`** to remove authentication:
   ```typescript
   const response = await fetch(API_UPLOAD_URL, {
     method: 'POST',
     // No Authorization header needed
     body: formData,
   });
   ```

3. **No `.env` file needed** - public spaces are accessible without tokens

### Testing the Deployment

#### Testing a Private Space

```bash
# Health check with token
curl -H "Authorization: Bearer hf_your_token_here" \
  https://yourusername-yourspacename.hf.space/health

# Test with a video file
curl -H "Authorization: Bearer hf_your_token_here" \
  -X POST \
  -F "video=@/path/to/test_video.mp4;type=video/mp4" \
  https://yourusername-yourspacename.hf.space/upload-video
```

#### Testing a Public Space

```bash
# Health check (no token needed)
curl https://yourusername-yourspacename.hf.space/health

# Test with a video file
curl -X POST \
  -F "video=@/path/to/test_video.mp4;type=video/mp4" \
  https://yourusername-yourspacename.hf.space/upload-video
```

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Pain Detection Server is running",
  "status": "ready",
  "model_type": "sta_lstm",
  "num_classes": 2,
  "endpoints": {
    "upload": "/upload-video",
    "health": "/health"
  }
}
```

### `GET /health`
Detailed server status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "config": {
    "model_type": "sta_lstm",
    "num_classes": 2,
    "num_features": 300,
    "max_sequence_length": 200,
    "use_frontalization": true,
    "compute_euclidean": false
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
  "painLevel": "VERY_STRONG_PAIN",
  "confidence": 0.8928694725036621,
  "numFrames": 46,
  "probabilities": {
    "NO_PAIN": 0.1071305125951767,
    "VERY_STRONG_PAIN": 0.8928694725036621
  },
  "description": "Very Strong Pain. Critical condition requiring immediate medical intervention."
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
