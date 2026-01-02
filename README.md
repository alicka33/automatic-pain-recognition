# Automatic Pain Recognition
**Bachelor thesis: Automatic pain recognition with the use of computer vision algorithms**

This project focuses on automatic pain recognition from facial expressions using deep learning models trained on facial landmark sequences extracted from video data. The main goal is to investigate whether temporal patterns of facial movements can be effectively used to detect the presence and intensity of pain.

The system supports multiple facial landmark detection pipelines (Dlib and MediaPipe), applies normalization and frontalization techniques, and trains sequential neural network models to perform binary and multi-class pain classification.

---

## Project Overview

The project is organized as a modular processing and training pipeline:
1. Raw video recordings are processed to detect faces and extract facial landmarks.
2. Landmark coordinates are normalized and optionally frontalized.
3. Numerical feature sequences (e.g. Euclidean distances between landmarks) are generated.
4. Deep learning models are trained on the extracted features to classify pain levels.
5. Training and evaluation results are visualized and stored for further analysis.

Due to the large size of video data and intermediate artifacts, datasets and trained models are stored on Google Drive and loaded dynamically during experiments.

---

## How to Run

### Environment setup
Install required dependencies:
```bash
pip install -r requirements.txt

```
### Project structure

```bash
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
│   └── frontalization/
│
├── data_preparation/
│   ├── process_dataset.py
│   ├── processing_pipeline_dlib.py
│   ├── processing_pipeline_mediapipe.py
│   │
│   ├── data_division/
│   │   ├── data_division_BioVid_HeatPain.ipynb
│   │   ├── data_division_BioVid_HeatPain.py
│   │   ├── data_division_RAVDESS.ipynb
│   │   └── data_division_RAVDESS.py
│   │
│   └── data_preparation/
│       ├── centering.ipynb
│       ├── data_preparation_BioVid_HeatPain.ipynb
│       ├── data_preparation_RAVDESS.ipynb
│       ├── dlib_mediapipe_comparison.ipynb
│       ├── face_detection.ipynb
│       ├── frontalization.ipynb
│       ├── landmark_detection.ipynb
│       └── video_to_frames.ipynb
│
├── models/
│   ├── Attention_LSTM.py
│   ├── Bi_LSTM.py
│   └── Transformer.py
│
├── training_utils/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocessed_dataset.py
│
├── pain_detection/
│   ├── training_pain.py
│   ├── evaluation_pain.py
│   │
│   └── training_pain/
│       ├── training_Attention_LSTM_binary.py
│       ├── training_Attention_LSTM_multiclass.py
│       ├── training_Bi_LSTM_binary.py
│       ├── training_Bi_LSTM_multiclass.py
│       ├── training_Transformer_binary.py
│       └── training_Transformer_multiclass.py
│
├── processing_pipeline_verification_on_emotions/
│   └── processed_data_verification/
│       ├── evaluation_emotion.py
│       ├── training_emotion.py
│       ├── average_landmark_movement_BioVid_HeatPain.ipynb
│       ├── average_landmark_movement_RAVDESS.ipynb
│       ├── landmark_verification_BioVid_HeatPain.ipynb
│       └── landmark_verification_RAVDESS.ipynb
│
├── pain_detection_app/
│
├── pain_detection_app_server/
│
├── tests/
│   ├── test_processing_pipeline_dlib.py
│   └── test_processing_pipeline_mediapipe.py
│
└── .gitignore
```