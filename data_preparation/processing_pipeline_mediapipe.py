import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Optional, Tuple, List

import mediapipe as mp
from mediapipe.python.solutions import face_mesh as face_mesh

# Constants (adapt as needed)
REFERENCE_POINT_INDEX = 2
NUM_LANDMARKS = 478
INPUT_FEATURE_DIM = NUM_LANDMARKS * 3
FRONTALIZATION_OUTPUT_SIZE = (300, 300)


def load_reference_keypoints(ref_path: Path, num_landmarks: int = NUM_LANDMARKS) -> Tuple[Optional[np.ndarray], bool]:
    """
    Load 3D reference keypoints (N,3). Returns (array or None, use_frontalization_flag).
    """
    try:
        ref = np.load(str(ref_path))
        if ref.ndim > 2:
            ref = ref.squeeze()
        if ref.shape == (num_landmarks, 3):
            return ref, True
        else:
            # shape mismatch
            return None, False
    except FileNotFoundError:
        return None, False
    except Exception:
        return None, False


def create_face_mesh(refine_landmarks: bool = True, max_num_faces: int = 1,
                     min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
    """
    Create a MediaPipe FaceMesh instance.
    """
    return face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=refine_landmarks,
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )


def parse_landmarks_from_results(results, expected_num: int = NUM_LANDMARKS) -> Optional[np.ndarray]:
    """
    Convert MediaPipe landmarks result to a (N,3) numpy array (x,y,z).
    Returns None if not found or wrong length.
    """
    if not results or not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    if coords.shape[0] != expected_num:
        return None
    return coords


def procrustes_analysis(X: np.ndarray, Y: np.ndarray, num_landmarks: int = NUM_LANDMARKS) -> np.ndarray:
    """
    Align X to Y using Procrustes-like analysis (3D). Expects shape (N,3).
    Returns aligned X in same coordinate scale as Y.
    """
    if X.shape != (num_landmarks, 3) or Y.shape != (num_landmarks, 3):
        raise ValueError("Procrustes input shapes mismatch.")

    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Normalize
    X_norm = np.linalg.norm(Xc)
    Y_norm = np.linalg.norm(Yc)
    if X_norm == 0 or Y_norm == 0:
        raise ValueError("Zero norm encountered in Procrustes.")

    Xc /= X_norm
    Yc /= Y_norm

    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ Vt

    X_aligned = Xc @ R
    X_aligned = X_aligned * Y_norm + Y_mean
    return X_aligned


def center_keypoints(keypoints: np.ndarray, reference_index: int = REFERENCE_POINT_INDEX) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subtract the reference point coordinates from all keypoints (translation centering).
    Returns (centered_keypoints (N,3), reference_coords (3,))
    """
    ref_coords = keypoints[reference_index].copy()
    centered = keypoints - ref_coords
    return centered, ref_coords


def keypoints_to_feature_vector(centered_keypoints: np.ndarray) -> np.ndarray:
    """
    Flatten centered (N,3) to a 1D float32 vector of length N*3.
    """
    return centered_keypoints.flatten().astype(np.float32)


def visualize_raw_detection(frame_bgr: np.ndarray, landmarks_list, reference_index: int = REFERENCE_POINT_INDEX):
    """
    Draw raw MediaPipe landmarks, face detection rectangle, and the reference point on the frame and show it.
    """
    h, w, _ = frame_bgr.shape
    img = frame_bgr.copy()

    # Draw face detection bounding box
    xs = [lm.x * w for lm in landmarks_list]
    ys = [lm.y * h for lm in landmarks_list]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Draw all landmarks
    for lm in landmarks_list:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)

    # Mark reference point
    ref_lm = landmarks_list[reference_index]
    rcx, rcy = int(ref_lm.x * w), int(ref_lm.y * h)
    cv2.circle(img, (rcx, rcy), 6, (0, 255, 0), -1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_frontalized_points(keypoints: np.ndarray, output_size: Tuple[int, int] = FRONTALIZATION_OUTPUT_SIZE, margin: int = 10):
    """
    Render frontalized 3D keypoints as a 2D scatter image (project X,Y).
    Returns the image for display.
    """
    canvas = np.ones((output_size[0], output_size[1], 3), dtype=np.uint8) * 255
    min_coords = keypoints[:, :2].min(axis=0)
    max_coords = keypoints[:, :2].max(axis=0)
    range_coords = max_coords - min_coords
    if np.any(range_coords == 0):
        range_coords = np.maximum(range_coords, 1e-6)
    scale = (output_size[0] - 2 * margin) / np.max(range_coords)

    # Center the points on the canvas
    center_offset = np.array([(output_size[0] - margin * 2) / 2, (output_size[1] - margin * 2) / 2])
    center_keypoints = (max_coords + min_coords) / 2

    for x, y, _ in keypoints:
        px = int((x - center_keypoints[0]) * scale) + output_size[0] // 2
        py = int((y - center_keypoints[1]) * scale) + output_size[1] // 2
        if 0 <= px < output_size[0] and 0 <= py < output_size[1]:
            cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1)

    # Draw grid lines thicker
    cv2.line(canvas, (margin, margin), (output_size[0] - margin, margin), (200, 200, 200), 2)
    cv2.line(canvas, (output_size[0] - margin, margin), (output_size[0] - margin, output_size[1] - margin), (200, 200, 200), 2)
    cv2.line(canvas, (output_size[0] - margin, output_size[1] - margin), (margin, output_size[1] - margin), (200, 200, 200), 2)
    cv2.line(canvas, (margin, output_size[1] - margin), (margin, margin), (200, 200, 200), 2)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def visualize_detections_side_by_side(frame_bgr: np.ndarray, landmarks_list, processed_keypoints: np.ndarray, reference_index: int = REFERENCE_POINT_INDEX):
    """
    Display raw detection and frontalized points side by side.
    """
    # Get raw detection image with bounding box
    raw_img = visualize_raw_detection(frame_bgr, landmarks_list, reference_index)

    # Get frontalized image
    frontal_img = visualize_frontalized_points(processed_keypoints)

    # Display side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(raw_img)
    axes[0].set_title('Raw Detection (478 landmarks + Face Box)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(frontal_img)
    axes[1].set_title('Frontalized Points (Centered)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()


def process_frame(frame_bgr: np.ndarray, face_mesh, reference_keypoints_3d: Optional[np.ndarray] = None,
                  use_frontalization: bool = False, reference_index: int = REFERENCE_POINT_INDEX) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a single BGR frame:
      - detect landmarks
      - optionally frontalize (align to reference)
      - center using reference point
      - return (feature_vector, processed_keypoints)
    Returns None if no valid detection.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    coords = parse_landmarks_from_results(results)
    if coords is None:
        return None

    processed = coords
    if use_frontalization and reference_keypoints_3d is not None:
        processed = procrustes_analysis(processed, reference_keypoints_3d)

    centered, _ = center_keypoints(processed, reference_index)
    feature_vector = keypoints_to_feature_vector(centered)
    return feature_vector, processed


def video_to_feature_sequences(video_path: str, frame_skip: int = 3,
                               reference_keypoints_3d: Optional[np.ndarray] = None,
                               use_frontalization: bool = False,
                               visualize: bool = False) -> List[np.ndarray]:
    """
    Process a video file into a list of feature vectors (one per processed frame).
    Uses the modular functions above so individual steps can be called in isolation.
    """
    sequences = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return sequences

    face_mesh = create_face_mesh()

    frame_count = 0
    visualization_done = False

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            try:
                out = process_frame(frame_bgr, face_mesh, reference_keypoints_3d,
                                    use_frontalization=use_frontalization,
                                    reference_index=REFERENCE_POINT_INDEX)
            except Exception as e:
                # Skip problematic frames
                frame_count += 1
                continue

            if out is None:
                frame_count += 1
                continue

            feature_vector, processed_keypoints = out
            sequences.append(feature_vector)

            if visualize and not visualization_done:
                # raw detection (requires original landmarks from MediaPipe results)
                # We call face_mesh.process again for the raw landmarks for display purposes
                results = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if results and results.multi_face_landmarks:
                    visualize_detections_side_by_side(frame_bgr, results.multi_face_landmarks[0].landmark, processed_keypoints, REFERENCE_POINT_INDEX)
                visualization_done = True

        frame_count += 1

    cap.release()
    face_mesh.close()
    return sequences
