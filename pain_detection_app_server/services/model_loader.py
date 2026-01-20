from typing import Optional, Tuple
import numpy as np
import torch
from pathlib import Path

import config
from model.Attention_LSTM import AttentionSequenceModel
from model.Bi_LSTM import SequenceModel
from model.STA_LSTM import STA_LSTM
from services.video_inference import VideoInferenceHelper, load_model_for_inference


class ModelLoader:
    """
    Handles loading and initialization of ML models and inference components.
    """

    def __init__(self):
        """Initialize the model loader."""
        self.model = None
        self.inference_helper = None
        self.device = None

    def load_assets(self) -> Tuple[bool, Optional[str]]:
        """Load model checkpoint and initialize inference pipeline."""
        try:
            print("Loading ML assets...")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            if not config.MODEL_CHECKPOINT_PATH.exists():
                error_msg = f"Model checkpoint not found at {config.MODEL_CHECKPOINT_PATH}"
                print(f"ERROR: {error_msg}")
                return False, error_msg

            model_class = self._get_model_class()

            self.model = load_model_for_inference(
                model_class=model_class,
                model_path=str(config.MODEL_CHECKPOINT_PATH),
                device=self.device,
                input_size=config.NUM_FEATURES,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                num_classes=config.NUM_CLASSES,
                dropout_prob=config.DROPOUT_PROB
            )
            print(f"Model loaded successfully from {config.MODEL_CHECKPOINT_PATH}")

            indices = self._load_landmark_indices()

            self.inference_helper = VideoInferenceHelper(
                model=self.model,
                device=self.device,
                indices=indices,
                compute_euclidean=config.COMPUTE_EUCLIDEAN,
                center_point_index=config.CENTER_POINT_INDEX,
                max_sequence_length=config.MAX_SEQUENCE_LENGTH,
                min_sequence_length=config.MIN_SEQUENCE_LENGTH,
                default_sequence_length=config.DEFAULT_SEQUENCE_LENGTH,
                use_adaptive_sequence_length=config.USE_ADAPTIVE_SEQUENCE_LENGTH,
                reference_keypoints_path=config.REFERENCE_KEYPOINTS_PATH if config.USE_FRONTALIZATION else None,
                use_frontalization=config.USE_FRONTALIZATION,
                frame_skip=config.FRAME_SKIP,
                min_frames_threshold=config.MIN_FRAMES_THRESHOLD
            )
            print("Inference pipeline initialized successfully")

            return True, None

        except Exception as e:
            error_msg = f"Error loading ML assets: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg

    def _get_model_class(self):
        """Get model class based on MODEL_TYPE config."""
        if config.MODEL_TYPE == "attention_lstm":
            return AttentionSequenceModel
        elif config.MODEL_TYPE == "bi_lstm":
            return SequenceModel
        elif config.MODEL_TYPE == "sta_lstm":
            return STA_LSTM
        else:
            raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

    def _load_landmark_indices(self) -> Optional[np.ndarray]:
        """Load landmark indices, auto-expand to xyz if needed."""
        if config.LANDMARK_INDICES_PATH.exists():
            indices = np.load(str(config.LANDMARK_INDICES_PATH))
            indices = np.array(indices, dtype=int)

            if not config.COMPUTE_EUCLIDEAN and config.NUM_FEATURES % 3 == 0:
                expected_points = config.NUM_FEATURES // 3
                if len(indices) == expected_points:
                    x = indices * 3
                    y = indices * 3 + 1
                    z = indices * 3 + 2
                    indices = np.stack([x, y, z], axis=1).flatten()
                    print(
                        "Expanded landmark indices to xyz coordinates "
                        f"({expected_points} points -> {len(indices)} features)"
                    )

            print(f"Loaded {len(indices)} landmark indices")
            return indices
        return None

    def is_loaded(self) -> bool:
        """Check if model and helper are loaded."""
        return self.model is not None and self.inference_helper is not None

    def get_inference_helper(self) -> Optional[VideoInferenceHelper]:
        """Get the inference helper instance."""
        return self.inference_helper

    def get_device_info(self) -> str:
        """Get device name (cpu/cuda)."""
        return str(self.device) if self.device else "unknown"
