# Automatic Pain Recognition

**Bachelor thesis: Application of computer vision algorithms for automatic pain recognition in patients**

Research project investigating automatic pain recognition from facial expressions using deep learning. The system analyzes temporal patterns of facial landmarks to detect presence and intensity of pain.

**Main Focus:** Data processing, feature engineering, and LSTM model training for pain classification.

---

## Research Overview

### Pipeline

```
Raw Video Data (BioVid, RAVDESS)
    ↓ (MediaPipe/Dlib landmark extraction)
Facial Landmark Sequences
    ↓ (processing: frontalization, centering)
Feature Engineering (Euclidean distances, selection)
    ↓ (train/val/test split)
LSTM Model Training (Bi-LSTM, Attention-LSTM, STA-LSTM)
    ↓
Pain Classification (binary & multiclass)
    ↓
Evaluation & Analysis
```

### Key Components

- **Data Preparation** - Extract facial landmarks from video datasets using MediaPipe/Dlib, frontalize, center
- **Feature Engineering** - Optionally compute Euclidean distances, select discriminative features
- **Model Training** - Train three LSTM architectures on processed sequences
- **Evaluation** - Assess model performance and analyze predictions

### Integration with Production

- **pain_detection_app_server/** - FastAPI server deployed on Hugging Face Spaces for inference
- **pain_detection_app/** - React Native mobile app (open in Android Studio) for video capture and upload

These are included as reference implementations but the main research focus is the training pipeline.

---

## Getting Started

### Setup in Google Colab

All experiments are designed for Google Colab. Clone the repository and run notebooks sequentially:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content')
!git clone https://github.com/alicka33/automatic-pain-recognition.git
os.chdir('/content/automatic-pain-recognition')

!pip install -r requirements.txt
```

### Detailed Structure

```
automatic-pain-recognition/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── datasets/
│   │   ├── BioVid_HeatPain_Database.md
│   │   └── RAVDESS_Database.md
│   ├── landmarks/
│   │   └── top_100_important_landmarks_emotions.npy
│   └── frontalization/
│       └── key_points_xyz.npy
│
├── data_preparation/
│   ├── process_dataset.py
│   ├── processing_pipeline_dlib.py
│   ├── processing_pipeline_mediapipe.py
│   ├── data_division/
│   │   ├── data_division_BioVid_HeatPain.ipynb
│   │   └── data_division_RAVDESS.ipynb
│   └── data_preparation/
│       ├── centering.ipynb
│       ├── data_preparation_BioVid_HeatPain.ipynb
│       ├── data_preparation_RAVDESS.ipynb
│       ├── dlib_mediapipe_comparison.ipynb
│       ├── face_detection.ipynb
│       ├── frontalization.ipynb
│       ├── video_to_frames.ipynb
│       └── video_to_landmarks_full_pipeline.ipynb
│
├── models/
│   ├── Attention_LSTM.py
│   ├── Bi_LSTM.py
│   ├── STA_LSTM.py
│   └── Emotion_Conv_LSTM.py
│
├── training_utils/
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocessed_dataset.py
│   └── __init__.py
│
├── pain_detection/
│   ├── analyse_data/
│   │   ├── average_landmark_movement_BioVid_HeatPain.ipynb
│   │   └── landmark_verification_BioVid_HeatPain.ipynb
│   └── training_pain/
│       ├── training_Attention_LSTM_*.ipynb
│       ├── training_Bi_LSTM_*.ipynb
│       ├── training_STA_LSTM_*.ipynb
│       └── ... (binary & multiclass variants)
│
├── processing_pipeline_verification_on_emotions/
│   ├── processed_data_verification/
│   └── training_emotion/
│
├── tests/
│   ├── test_process_dataset_colab.ipynb
│   ├── test_processing_pipeline_dlib_colab.ipynb
│   ├── test_processing_pipeline_mediapipe_colab.ipynb
│   └── test_training_utils_colab.ipynb
│
├── pain_detection_app_server/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── app.py
│   ├── config.py
│   ├── Dockerfile
│   ├── services/
│   ├── model/
│   └── tests/
│
├── pain_detection_app/
│   ├── README.md
│   ├── app/
│   ├── components/
│   ├── services/
│   └── constants/
│
└── .gitignore
```

---

## Production Deployment

### Server (Inference)

The trained models can be deployed via the FastAPI server on **Hugging Face Spaces**:

```bash
cd pain_detection_app_server
# See pain_detection_app_server/README.md for Hugging Face deployment
```

Server provides REST API for video upload and pain classification.

### Mobile App

Built with React Native/Expo for Android/iOS. Open project in **Android Studio**:

```bash
cd pain_detection_app
# See pain_detection_app/README.md for Android Studio setup
```

Connect to the deployed server for real-time pain detection.

---

## Data Sources & References

### Facial Landmarks

The facial landmark selection strategy is based on:

**Source:** [Newtoneiro/automatic-lie-detection](https://github.com/Newtoneiro/automatic-lie-detection)
- Master's thesis research by B. Latoszek (2025)
- Warsaw University of Technology

**Landmark Files in `data/landmarks/`:**
- `top_100_important_landmarks_emotions.npy`
- `combined_selected_points_emotions.npy`
- `manualy_selected_points.npy`

### Frontalization Methods

Face frontalization normalizes facial landmarks to a canonical frontal view using Procrustes alignment.

**MediaPipe Frontalization:**
- Reference canonical keypoints: `data/frontalization/key_points_xyz.npy`
- Source: [Newtoneiro/automatic-lie-detection](https://github.com/Newtoneiro/automatic-lie-detection)
- Used in: `data_preparation/processing_pipeline_mediapipe.py`

**Dlib Frontalization:**
- Implementation: [bbonik/facial-landmark-frontalization](https://github.com/bbonik/facial-landmark-frontalization)
- Pre-trained models: [davisking/dlib-models](https://github.com/davisking/dlib-models)
- Used in: `data_preparation/processing_pipeline_dlib.py`

---
