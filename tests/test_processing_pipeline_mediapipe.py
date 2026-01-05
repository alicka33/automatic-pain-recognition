import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from data_preparation.processing_pipeline_mediapipe import (
    load_reference_keypoints,
    create_face_mesh,
    parse_landmarks_from_results,
    procrustes_analysis,
    center_keypoints,
    keypoints_to_feature_vector,
    process_frame,
)


@pytest.fixture
def dummy_reference_keypoints():
    """Generate synthetic reference keypoints (478, 3)."""
    return np.random.randn(478, 3).astype(np.float32)


@pytest.fixture
def dummy_frame():
    """Generate a dummy BGR frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def temp_npy_file(dummy_reference_keypoints):
    """Create a temporary .npy file with reference keypoints."""
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, dummy_reference_keypoints)
        yield Path(f.name)
    Path(f.name).unlink()


# ==================== Tests: load_reference_keypoints ====================
def test_load_reference_keypoints_success(temp_npy_file, dummy_reference_keypoints):
    """Test loading valid reference keypoints."""
    ref, use_front = load_reference_keypoints(temp_npy_file, num_landmarks=478)
    assert ref is not None
    assert ref.shape == (478, 3)
    assert np.allclose(ref, dummy_reference_keypoints)
    assert use_front is True


def test_load_reference_keypoints_file_not_found():
    """Test handling of missing file."""
    ref, use_front = load_reference_keypoints(Path('/nonexistent/path.npy'), num_landmarks=478)
    assert ref is None
    assert use_front is False


def test_load_reference_keypoints_shape_mismatch(temp_npy_file):
    """Test handling of shape mismatch."""
    # Save wrong shape
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, np.random.randn(100, 3))
        wrong_path = Path(f.name)

    ref, use_front = load_reference_keypoints(wrong_path, num_landmarks=478)
    assert ref is None
    assert use_front is False
    wrong_path.unlink()


# ==================== Tests: create_face_mesh ====================
def test_create_face_mesh():
    """Test FaceMesh creation."""
    mesh = create_face_mesh(refine_landmarks=True, max_num_faces=1)
    assert mesh is not None
    # Verify it has process method
    assert hasattr(mesh, 'process')


# ==================== Tests: parse_landmarks_from_results ====================
def test_parse_landmarks_from_results_success():
    """Test parsing valid landmarks."""
    mock_landmark = Mock()
    mock_landmark.x, mock_landmark.y, mock_landmark.z = 0.5, 0.5, 0.1

    mock_face_landmarks = Mock()
    mock_face_landmarks.landmark = [mock_landmark] * 478

    mock_results = Mock()
    mock_results.multi_face_landmarks = [mock_face_landmarks]

    coords = parse_landmarks_from_results(mock_results, expected_num=478)
    assert coords is not None
    assert coords.shape == (478, 3)
    assert np.allclose(coords[0], [0.5, 0.5, 0.1])


def test_parse_landmarks_from_results_no_detection():
    """Test parsing when no face detected."""
    mock_results = Mock()
    mock_results.multi_face_landmarks = None

    coords = parse_landmarks_from_results(mock_results, expected_num=478)
    assert coords is None


def test_parse_landmarks_from_results_wrong_count():
    """Test parsing with wrong landmark count."""
    mock_landmark = Mock()
    mock_landmark.x, mock_landmark.y, mock_landmark.z = 0.5, 0.5, 0.1

    mock_face_landmarks = Mock()
    mock_face_landmarks.landmark = [mock_landmark] * 100  # Wrong count

    mock_results = Mock()
    mock_results.multi_face_landmarks = [mock_face_landmarks]

    coords = parse_landmarks_from_results(mock_results, expected_num=478)
    assert coords is None


# ==================== Tests: procrustes_analysis ====================
def test_procrustes_analysis_success(dummy_reference_keypoints):
    """Test Procrustes alignment."""
    X = np.random.randn(478, 3).astype(np.float32)
    Y = dummy_reference_keypoints

    X_aligned = procrustes_analysis(X, Y, num_landmarks=478)
    assert X_aligned.shape == (478, 3)
    assert not np.any(np.isnan(X_aligned))


def test_procrustes_analysis_shape_mismatch():
    """Test Procrustes with shape mismatch."""
    X = np.random.randn(100, 3)
    Y = np.random.randn(478, 3)

    with pytest.raises(ValueError, match="shape mismatch"):
        procrustes_analysis(X, Y, num_landmarks=478)


def test_procrustes_analysis_zero_norm():
    """Test Procrustes with zero norm (should raise)."""
    X = np.zeros((478, 3))
    Y = np.random.randn(478, 3)

    with pytest.raises(ValueError, match="Zero norm"):
        procrustes_analysis(X, Y, num_landmarks=478)


# ==================== Tests: center_keypoints ====================
def test_center_keypoints_success():
    """Test centering keypoints around a reference."""
    keypoints = np.random.randn(478, 3).astype(np.float32)
    ref_idx = 2

    centered, ref_coords = center_keypoints(keypoints, reference_index=ref_idx)

    assert centered.shape == (478, 3)
    assert ref_coords.shape == (3,)
    assert np.allclose(centered[ref_idx], 0.0)  # Reference point should be at origin
    assert np.allclose(ref_coords, keypoints[ref_idx])


def test_center_keypoints_out_of_bounds():
    """Test centering with invalid reference index (edge case)."""
    keypoints = np.random.randn(478, 3).astype(np.float32)
    ref_idx = 10000  # Out of bounds

    # Should raise IndexError when accessing keypoints[ref_idx]
    with pytest.raises(IndexError):
        center_keypoints(keypoints, reference_index=ref_idx)


# ==================== Tests: keypoints_to_feature_vector ====================
def test_keypoints_to_feature_vector_success():
    """Test flattening keypoints to feature vector."""
    keypoints = np.random.randn(478, 3).astype(np.float32)
    feature = keypoints_to_feature_vector(keypoints)

    assert feature.shape == (478 * 3,)
    assert feature.dtype == np.float32
    assert np.allclose(feature, keypoints.flatten())


def test_keypoints_to_feature_vector_shape():
    """Test feature vector shape for different landmark counts."""
    for n_landmarks in [68, 478]:
        keypoints = np.random.randn(n_landmarks, 3).astype(np.float32)
        feature = keypoints_to_feature_vector(keypoints)
        assert feature.shape == (n_landmarks * 3,)


# ==================== Tests: process_frame ====================
@patch('data_preparation.processing_pipeline_mediapipe.create_face_mesh')
@patch('data_preparation.processing_pipeline_mediapipe.parse_landmarks_from_results')
def test_process_frame_success(mock_parse, mock_mesh, dummy_frame, dummy_reference_keypoints):
    """Test successful frame processing."""
    # Mock landmarks
    mock_coords = np.random.randn(478, 3).astype(np.float32)
    mock_parse.return_value = mock_coords

    mock_face_mesh = Mock()
    mock_mesh.return_value = mock_face_mesh
    mock_face_mesh.process.return_value = Mock()

    result = process_frame(dummy_frame, mock_face_mesh, reference_keypoints_3d=dummy_reference_keypoints, 
                          use_frontalization=False, reference_index=2)

    assert result is not None
    feature_vector, processed_keypoints = result
    assert feature_vector.shape == (478 * 3,)
    assert processed_keypoints.shape == (478, 3)


@patch('data_preparation.processing_pipeline_mediapipe.create_face_mesh')
@patch('data_preparation.processing_pipeline_mediapipe.parse_landmarks_from_results')
def test_process_frame_no_detection(mock_parse, mock_mesh, dummy_frame):
    """Test frame processing with no face detected."""
    mock_parse.return_value = None  # No detection

    mock_face_mesh = Mock()
    result = process_frame(dummy_frame, mock_face_mesh, use_frontalization=False)

    assert result is None


# ==================== Integration Tests ====================
def test_pipeline_chain(dummy_reference_keypoints):
    """Test chaining multiple functions together."""
    # Create synthetic keypoints
    keypoints = np.random.randn(478, 3).astype(np.float32)

    # Center them
    centered, ref_coords = center_keypoints(keypoints, reference_index=2)

    # Convert to feature vector
    feature = keypoints_to_feature_vector(centered)

    # Verify chain
    assert feature.shape == (478 * 3,)
    assert feature.dtype == np.float32
    assert np.allclose(centered[2], 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
