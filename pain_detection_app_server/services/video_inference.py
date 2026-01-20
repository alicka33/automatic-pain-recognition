import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

from services.processing_pipeline_mediapipe import (
    video_to_feature_sequences,
    load_reference_keypoints,
)


class VideoInferenceHelper:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        indices: Optional[np.ndarray] = None,
        compute_euclidean: bool = True,
        center_point_index: int = 2,
        num_coords_per_point: int = 3,
        max_sequence_length: int = 46,
        min_sequence_length: int = 10,
        default_sequence_length: int = 46,
        use_adaptive_sequence_length: bool = True,
        pad_value: float = 0.0,
        reference_keypoints_path: Optional[Path] = None,
        use_frontalization: bool = False,
        frame_skip: int = 3,
        min_frames_threshold: int = 5,
    ):
        self.model = model
        self.device = device
        self.indices = np.array(indices, dtype=int) if indices is not None else None
        self.compute_euclidean = compute_euclidean
        self.center_point_index = int(center_point_index)
        self.num_coords_per_point = int(num_coords_per_point)
        self.max_sequence_length = int(max_sequence_length)
        self.min_sequence_length = int(min_sequence_length)
        self.default_sequence_length = int(default_sequence_length)
        self.use_adaptive_sequence_length = use_adaptive_sequence_length
        self.pad_value = float(pad_value)
        self.frame_skip = frame_skip
        self.min_frames_threshold = min_frames_threshold

        self.reference_keypoints_3d = None
        self.use_frontalization = use_frontalization
        if use_frontalization and reference_keypoints_path is not None:
            ref_kp, success = load_reference_keypoints(reference_keypoints_path)
            if success and ref_kp is not None:
                self.reference_keypoints_3d = ref_kp
                self.use_frontalization = True
            else:
                print("Warning: Failed to load reference keypoints, disabling frontalization")
                self.use_frontalization = False

        self.model.eval()

    def _compute_euclidean(self, seq_3d: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances from center point."""
        T, F_full = seq_3d.shape
        if F_full % self.num_coords_per_point != 0:
            raise ValueError(
                f"Invalid shape: F_full ({F_full}) not divisible by "
                f"num_coords_per_point ({self.num_coords_per_point})"
            )

        N_points = F_full // self.num_coords_per_point
        points = seq_3d.reshape(T, N_points, self.num_coords_per_point)

        if not (0 <= self.center_point_index < N_points):
            raise IndexError(
                f"center_point_index {self.center_point_index} out of range (0..{N_points-1})"
            )

        center = points[:, self.center_point_index, :]
        center_rep = center[:, None, :]
        diff = points - center_rep
        dists = np.linalg.norm(diff, axis=2)
        return dists

    def _select_features(self, features: np.ndarray) -> np.ndarray:
        """Select feature subset using indices if provided."""
        if self.indices is not None:
            if len(self.indices) > 0 and max(self.indices) >= features.shape[1]:
                raise IndexError("Provided indices contain index >= feature dimension.")
            return features[:, self.indices]
        return features

    def _pad_or_truncate(self, arr: np.ndarray) -> np.ndarray:
        """Adaptively handle sequence length based on configuration."""
        T, F = arr.shape

        if self.use_adaptive_sequence_length:
            if T > self.max_sequence_length:
                print(f"Video has {T} frames, truncating to {self.max_sequence_length}")
                return arr[:self.max_sequence_length]
            elif T < self.min_sequence_length:
                pad_shape = (self.min_sequence_length - T, F)
                pad = np.full(pad_shape, self.pad_value, dtype=arr.dtype)
                print(f"Video has {T} frames, padding to {self.min_sequence_length}")
                return np.vstack([arr, pad])
            else:
                print(f"Using adaptive sequence length: {T} frames")
                return arr
        else:
            target_length = self.default_sequence_length
            if T == target_length:
                return arr
            if T < target_length:
                pad_shape = (target_length - T, F)
                pad = np.full(pad_shape, self.pad_value, dtype=arr.dtype)
                return np.vstack([arr, pad])
            return arr[:target_length]

    def process_video(self, video_path: str) -> Tuple[Optional[torch.Tensor], int]:
        """Process video into feature tensor for model."""
        feature_sequences = video_to_feature_sequences(
            video_path,
            frame_skip=self.frame_skip,
            reference_keypoints_3d=self.reference_keypoints_3d,
            use_frontalization=self.use_frontalization,
            visualize=False,
        )

        if len(feature_sequences) < self.min_frames_threshold:
            print(
                f"Warning: Only {len(feature_sequences)} valid frames detected (min: {self.min_frames_threshold})"
            )
            return None, len(feature_sequences)

        seq = np.stack(feature_sequences, axis=0).astype(np.float32)

        features = (
            self._compute_euclidean(seq) if self.compute_euclidean else seq
        )

        features = self._select_features(features)
        features = self._pad_or_truncate(features)

        sequence_tensor = torch.from_numpy(features.astype(np.float32))
        sequence_tensor = sequence_tensor.unsqueeze(0)

        return sequence_tensor, len(feature_sequences)

    def predict(self, video_path: str):
        """Run pain classification on video."""
        sequence_tensor, num_frames = self.process_video(video_path)
        if sequence_tensor is None:
            return None, None, num_frames

        with torch.no_grad():
            sequence_tensor = sequence_tensor.to(self.device)
            logits = self.model(sequence_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            probs_numpy = probabilities.cpu().numpy()[0]

        return predicted_class, probs_numpy, num_frames


def load_model_for_inference(
    model_class,
    model_path: str,
    device: torch.device,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    num_classes: int,
    dropout_prob: float = 0.3,
) -> torch.nn.Module:
    """Load model from checkpoint for inference."""
    model = model_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_prob=dropout_prob,
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model
