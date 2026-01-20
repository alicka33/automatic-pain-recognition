from typing import Dict, Any, Optional
import config
from services.video_inference import VideoInferenceHelper


class PainDetectionService:
    """
    Service for handling pain detection logic.

    This class encapsulates the business logic for analyzing videos
    and returning pain classification results.
    """

    def __init__(self, inference_helper: VideoInferenceHelper):
        self.inference_helper = inference_helper

    def analyze_video(self, file_path: str) -> Dict[str, Any]:
        """Analyze video and return pain classification results."""
        predicted_class, probabilities, num_frames = self.inference_helper.predict(file_path)

        if predicted_class is None:
            pain_level = "NO_FACE_DETECTED"
            confidence = 0.0
            prob_dict = {}
        else:
            pain_level = config.PAIN_CLASSES.get(predicted_class, "ERROR_ANALYSIS")
            confidence = float(probabilities[predicted_class])

            prob_dict = {
                config.PAIN_CLASSES[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }

        description = config.PAIN_DESCRIPTIONS.get(
            pain_level,
            "Unknown analysis result."
        )

        return {
            "painLevel": pain_level,
            "confidence": confidence,
            "numFrames": num_frames,
            "probabilities": prob_dict,
            "description": description
        }

    def get_service_info(self) -> Dict[str, Any]:
        """Get service configuration details."""
        return {
            "model_type": config.MODEL_TYPE,
            "num_classes": config.NUM_CLASSES,
            "num_features": config.NUM_FEATURES,
            "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
            "use_frontalization": config.USE_FRONTALIZATION,
            "compute_euclidean": config.COMPUTE_EUCLIDEAN
        }
