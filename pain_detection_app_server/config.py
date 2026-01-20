from pathlib import Path


SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

MODEL_CHECKPOINT_PATH = MODEL_DIR / "model_paths/testing_new_code_sta_lstm_2_classes_300_coord.pt"
REFERENCE_KEYPOINTS_PATH = DATA_DIR / "frontalization" / "key_points_xyz.npy"
LANDMARK_INDICES_PATH = DATA_DIR / "landmarks" / "top_100_important_landmarks_emotions.npy"

NUM_FEATURES = 300

HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 2
DROPOUT_PROB = 0.3

USE_ADAPTIVE_SEQUENCE_LENGTH = True
MIN_SEQUENCE_LENGTH = 10  # Minimum frames required for meaningful prediction
MAX_SEQUENCE_LENGTH = 200  # Maximum to prevent memory issues
DEFAULT_SEQUENCE_LENGTH = 46  # Fallback/padding target for short videos

CENTER_POINT_INDEX = 2


FRAME_SKIP = 3
MIN_FRAMES_THRESHOLD = 5
COMPUTE_EUCLIDEAN = False
USE_FRONTALIZATION = True


# Binary classification (0-1)
PAIN_CLASSES_BINARY = {
    0: "NO_PAIN",
    1: "VERY_STRONG_PAIN"
}

# Multiclass classification (0-4)
PAIN_CLASSES_MULTICLASS = {
    0: "NO_PAIN",
    1: "WEAK_PAIN",
    2: "MODERATE_PAIN",
    3: "STRONG_PAIN",
    4: "VERY_STRONG_PAIN"
}

PAIN_CLASSES = PAIN_CLASSES_BINARY if NUM_CLASSES == 2 else PAIN_CLASSES_MULTICLASS

PAIN_DESCRIPTIONS = {
    "NO_PAIN": "No Pain Detected. No signs of discomfort observed.",
    "PAIN": "Pain Detected. Signs of discomfort present.",
    "WEAK_PAIN": "Weak Pain. Minimal discomfort, basic monitoring recommended.",
    "MODERATE_PAIN": "Moderate Pain. Clear discomfort, consultation and assessment advised.",
    "STRONG_PAIN": "Strong Pain. Urgent intervention required to alleviate discomfort.",
    "VERY_STRONG_PAIN": "Very Strong Pain. Critical condition requiring immediate medical intervention.",
    "NO_FACE_DETECTED": "Detection Error: Insufficient face detection in video frames.",
    "ERROR_ANALYSIS": "Internal Error: Problem with model or data processing."
}

CONFIDENCE_THRESHOLD = 0.5

# Options: "attention_lstm", "bi_lstm", "sta_lstm"
MODEL_TYPE = "sta_lstm"
