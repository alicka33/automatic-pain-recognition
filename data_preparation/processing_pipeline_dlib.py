import os
import math
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
from imutils.face_utils import FaceAligner, rect_to_bb


def init_dlib(predictor_path: str, mean_face_path: Optional[str] = None, weights_path: Optional[str] = None,
              desired_face_size: int = 128) -> Dict:
    """
    Initialize dlib detector/predictor/aligner and optionally load frontalization resources.
    Returns dict with keys: detector, predictor, aligner, canonical_reference, frontalization_weights
    """
    ctx = {}
    try:
        ctx['detector'] = dlib.get_frontal_face_detector()
        ctx['predictor'] = dlib.shape_predictor(str(predictor_path))
        ctx['aligner'] = FaceAligner(ctx['predictor'], desiredFaceWidth=desired_face_size,
                                     desiredFaceHeight=desired_face_size)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load dlib predictor or init FaceAligner: {predictor_path}") from e

    try:
        ctx['canonical_reference'] = np.load(str(mean_face_path)).astype(np.float32) if mean_face_path else None
    except Exception:
        ctx['canonical_reference'] = None

    try:
        ctx['frontalization_weights'] = np.load(str(weights_path)).astype(np.float32) if weights_path else None
    except Exception:
        ctx['frontalization_weights'] = None

    return ctx


def landmark_obj_to_array(landmarks_dlib_obj) -> np.ndarray:
    """Convert dlib landmark object to (68, 2) numpy array."""
    arr = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        arr[i, 0] = landmarks_dlib_obj.part(i).x
        arr[i, 1] = landmarks_dlib_obj.part(i).y
    return arr


def landmark_vector_to_matrix(vec: np.ndarray) -> np.ndarray:
    """Reshape flattened landmark vector into (N, 2) coordinate matrix."""
    mid = len(vec) // 2
    return np.vstack((vec[:mid], vec[mid:])).T


def get_eye_centers(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute left and right eye center positions from landmarks."""
    left = landmarks[36:42].mean(axis=0)
    right = landmarks[42:48].mean(axis=0)
    return left, right


def procrustes_normalize(landmarks: np.ndarray, template: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize landmarks via Procrustes alignment, optionally anchored to template."""
    L = landmarks.copy().astype(np.float32)
    L -= L.mean(axis=0)
    scale = math.sqrt(np.mean(np.sum(L**2, axis=1)))
    if scale == 0:
        raise ValueError("Zero scale in procrustes normalization")
    L /= scale
    left, right = get_eye_centers(L)
    dx = right[0] - left[0]; dy = right[1] - left[1]
    angle = 0.0 if dx == 0 else math.atan2(dy, dx)
    R = np.array([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])
    L = L.dot(R.T)
    t = template
    if t is not None:
        anchor_t = t[50:53].mean(axis=0); anchor_l = L[50:53].mean(axis=0); L[48:] += (anchor_t - anchor_l)
        anchor_t = t[42:48].mean(axis=0); anchor_l = L[42:48].mean(axis=0); disp = anchor_t - anchor_l
        L[42:48] += disp; L[22:27] += disp
        anchor_t = t[36:42].mean(axis=0); anchor_l = L[36:42].mean(axis=0); disp = anchor_t - anchor_l
        L[36:42] += disp; L[17:22] += disp
        anchor_t = t[27:36].mean(axis=0); anchor_l = L[27:36].mean(axis=0); L[27:36] += (anchor_t - anchor_l)
        anchor_t = t[:17].mean(axis=0); anchor_l = L[:17].mean(axis=0); L[:17] += (anchor_t - anchor_l)
    return L


def frontalize_landmarks(landmarks_dlib_obj, frontalization_weights: Optional[np.ndarray],
                         canonical_reference: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Apply frontalization to landmarks using pre-trained weights and reference."""
    if frontalization_weights is None:
        return None
    L = landmark_obj_to_array(landmarks_dlib_obj)
    L_norm = procrustes_normalize(L, template=canonical_reference)
    landmark_vector = np.hstack((L_norm[:, 0].T, L_norm[:, 1].T, 1.0)).astype(np.float32)
    out_vec = landmark_vector.dot(frontalization_weights)
    return landmark_vector_to_matrix(out_vec).astype(np.float32)


def to_feature_vector(coords: np.ndarray) -> np.ndarray:
    """Flatten coordinate matrix to 1D feature vector."""
    return coords.flatten().astype(np.float32)


def center_by_reference(coords: np.ndarray, ref_index: int = 33) -> Tuple[np.ndarray, np.ndarray]:
    """Center coordinates by subtracting reference landmark position."""
    ref = coords[ref_index].copy()
    centered = coords - ref
    return centered, ref


def process_frame(frame_bgr: np.ndarray, detector, predictor, aligner=None,
                  frontalization_weights: Optional[np.ndarray] = None,
                  canonical_reference: Optional[np.ndarray] = None,
                  frontalize: bool = True, center_ref: bool = True, ref_index: int = 33,
                  visualize: bool = False) -> Optional[Dict]:
    """Detect face and extract landmarks from frame, returning raw/frontal/centered coordinates."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return None
    rect = rects[0]
    x, y, w, h = rect_to_bb(rect)
    rect_int = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    try:
        shape = predictor(gray, rect_int)
    except Exception:
        return None
    raw = landmark_obj_to_array(shape)
    frontal = None
    if frontalize and frontalization_weights is not None:
        try:
            frontal = frontalize_landmarks(shape, frontalization_weights, canonical_reference)
        except Exception:
            frontal = None
    used = frontal if frontal is not None else raw
    centered = None; feature_vector = None
    if used is not None:
        if center_ref:
            centered, _ = center_by_reference(used, ref_index)
            feature_vector = to_feature_vector(centered)
        else:
            centered = used; feature_vector = to_feature_vector(used)
    if visualize:
        _visualize_frame(frame_bgr, raw, frontal, rect=(x, y, w, h))
    return {
        'raw_landmarks': raw,
        'frontal_landmarks': frontal,
        'centered_coords': centered,
        'feature_vector': feature_vector
    }


def video_to_landmark_vectors(video_path: str, detector, predictor, aligner=None,
                              frontalization_weights: Optional[np.ndarray] = None,
                              canonical_reference: Optional[np.ndarray] = None,
                              frame_skip: int = 5, frontalize: bool = True, center_ref: bool = True,
                              ref_index: int = 33, visualize: bool = False) -> List[np.ndarray]:
    """Extract landmark feature vectors from video frames."""
    seq = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return seq
    frame_idx = 0; viz_done = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            out = process_frame(frame, detector, predictor, aligner, frontalization_weights,
                                canonical_reference, frontalize, center_ref, ref_index, visualize=(visualize and not viz_done))
            if out and out.get('feature_vector') is not None:
                seq.append(out['feature_vector'])
                if visualize and not viz_done:
                    viz_done = True
        frame_idx += 1
    cap.release()
    return seq


def _visualize_frame(frame_bgr: np.ndarray, raw: np.ndarray, frontal: Optional[np.ndarray], rect: Optional[Tuple[int,int,int,int]] = None):
    """Display raw and frontalized landmarks side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # raw overlay
    img_raw = frame_bgr.copy()
    if rect:
        x, y, w, h = rect
        cv2.rectangle(img_raw, (x, y), (x+w, y+h), (0, 255, 0), 3)
    for (px, py) in raw.astype(int):
        cv2.circle(img_raw, (px, py), 4, (0, 255, 0), -1)
    axes[0].imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)); axes[0].axis('off'); axes[0].set_title('Raw Landmarks')
    if frontal is not None:
        target_size = 256
        display = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        minc, maxc = frontal[:, :2].min(axis=0), frontal[:, :2].max(axis=0)
        rng = maxc - minc; rng[rng == 0] = 1.0
        scale = (target_size - 20) / max(rng)
        coords = (frontal * scale) + (target_size // 2)
        for (px, py) in coords.astype(int):
            if 0 <= px < target_size and 0 <= py < target_size:
                cv2.circle(display, (px, py), 2, (0, 255, 0), -1)
        axes[1].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)); axes[1].axis('off'); axes[1].set_title('Frontalized Points')
    else:
        axes[1].axis('off'); axes[1].set_title('No frontalization available')
    plt.show()
