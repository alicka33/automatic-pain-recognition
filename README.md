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

## Environment Setup

All experiments were designed to be executed in the Google Colab environment.
To begin, open the appropriate notebook in Google Colab and run the cells sequentially as provided.

If the notebook requires additional environment configuration (e.g., access to Google Drive, repository cloning, or installation of required libraries), all necessary steps are explicitly specified within the notebook itself. These steps typically include:

mounting Google Drive to access the dataset,

cloning the project repository,

installing dependencies from the requirements.txt file.

No additional manual configuration is required from the user beyond executing the code in the notebook cells.


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
│   │   └── data_division_RAVDESS.ipynb
│   │
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
│   ├── Emotion_Conv_LSTM.py
│   └── STA_LSTM.py
│
├── training_utils/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocessed_dataset.py
│
├── pain_detection/
│   ├── analyse_data/
│   │   ├── average_landmark_movement_BioVid_HeatPain.ipynb
│   │   └── landmark_verification_BioVid_HeatPain.ipynb
│   │
│   └── training_pain/
│       ├── training_Attention_LSTM_binary.ipynb
│       ├── training_Attention_LSTM_binary_1434_coord.ipynb
│       ├── training_Attention_LSTM_binary_300_coord.ipynb
│       ├── training_Attention_LSTM_multiclass.ipynb
│       ├── training_Attention_LSTM_multiclass_1434_coord.ipynb
│       ├── training_Attention_LSTM_multiclass_300_coord.ipynb
│       ├── training_Bi_LSTM_binary.ipynb
│       ├── training_Bi_LSTM_binary_1434_coord.ipynb
│       ├── training_Bi_LSTM_binary_300_coord.ipynb
│       ├── training_Bi_LSTM_multiclass.ipynb
│       ├── training_Bi_LSTM_multiclass_1434_coord.ipynb
│       ├── training_Bi_LSTM_multiclass_300_coord.ipynb
│       ├── training_STA_LSTM_binary.ipynb
│       ├── training_STA_LSTM_binary_1434_coord.ipynb
│       ├── training_STA_LSTM_binary_300_coord.ipynb
│       ├── training_STA_LSTM_multiclass.ipynb
│       ├── training_STA_LSTM_multiclass_1434_coord.ipynb
│       └── training_STA_LSTM_multiclass_300_coord.ipynb
│
├── processing_pipeline_verification_on_emotions/
│   ├── processed_data_verification/
│   └── training_emotion/
│
├── pain_detection_app/
│
├── pain_detection_app_server/
│
├── tests/
│   ├── test_process_dataset_colab.ipynb
│   ├── test_processing_pipeline_dlib_colab.ipynb
│   ├── test_processing_pipeline_mediapipe_colab.ipynb
│   └── test_training_utils_colab.ipynb
│
└── .gitignore
```