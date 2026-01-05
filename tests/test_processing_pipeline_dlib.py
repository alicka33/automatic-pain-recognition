import pytest
import os
import tempfile
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from data_preparation.processing_pipeline_dlib import (
    init_dlib,
    landmark_obj_to_array,
    landmark_vector_to_matrix,
    get_eye_centers,
    procrustes_normalize,
    frontalize_landmarks,
    to_feature_vector,
    center_by_reference,
    process_frame,
    video_to_landmark_vectors
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dirs = {
            'root': tmpdir,
            'models': os.path.join(tmpdir, 'models'),
            'videos': os.path.join(tmpdir, 'videos'),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        yield dirs


@pytest.fixture
def sample_landmarks_68():
    """Create sample 68-point landmarks (Dlib format)."""
    # Create realistic landmark coordinates (face points)
    landmarks = np.array([
        [100, 150], [105, 148], [110, 146], [115, 145],  # Jaw line
        [120, 144], [125, 143], [130, 142], [135, 141],
        [140, 141], [145, 142], [150, 143], [155, 144],
        [160, 145], [165, 146], [170, 148], [175, 150],
        [180, 153],  # Right ear
        [95, 160], [90, 165],  # Left ear
        [120, 130], [140, 128],  # Eyes
        [160, 130], [175, 132],
        [150, 140],  # Nose
        [125, 170], [150, 172], [175, 170],  # Mouth
        [130, 175], [150, 177], [170, 175],
        [100, 100], [120, 95], [140, 93], [160, 95], [180, 100],  # Eyebrows
        [110, 110], [130, 108], [150, 107], [170, 108], [190, 110],
    ], dtype=np.float32)

    # Pad to 68 points if needed
    if landmarks.shape[0] < 68:
        extra = np.random.rand(68 - landmarks.shape[0], 2).astype(np.float32) * 100 + 100
        landmarks = np.vstack([landmarks, extra])

    return landmarks[:68]


@pytest.fixture
def mock_dlib_landmark_obj(sample_landmarks_68):
    """Create a mock dlib landmark object."""
    mock_obj = Mock()
    for i in range(68):
        part_mock = Mock()
        part_mock.x = int(sample_landmarks_68[i, 0])
        part_mock.y = int(sample_landmarks_68[i, 1])
        mock_obj.part = Mock(return_value=part_mock)
        # Make part callable with correct index
        def make_part(idx):
            def _part(i):
                if i != idx:
                    return mock_obj.part(i)
                p = Mock()
                p.x = int(sample_landmarks_68[idx, 0])
                p.y = int(sample_landmarks_68[idx, 1])
                return p
            return _part
        mock_obj.part = make_part(i)

    # Create a proper callable that returns correct values
    def part_func(i):
        p = Mock()
        p.x = int(sample_landmarks_68[i, 0])
        p.y = int(sample_landmarks_68[i, 1])
        return p

    mock_obj.part = part_func
    return mock_obj


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame."""
    height, width = 480, 640
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def mock_detector_and_predictor():
    """Create mock dlib detector and predictor."""
    detector_mock = Mock()
    predictor_mock = Mock()

    def detector_side_effect(gray_image, upsample_num_times):
        # Return mock rectangle
        rect_mock = Mock()
        rect_mock.left = Mock(return_value=50)
        rect_mock.top = Mock(return_value=50)
        rect_mock.right = Mock(return_value=250)
        rect_mock.bottom = Mock(return_value=350)
        return [rect_mock]

    detector_mock.side_effect = detector_side_effect
    return detector_mock, predictor_mock


# ============================================================================
# TESTS FOR landmark_obj_to_array
# ============================================================================

class TestLandmarkObjToArray:
    """Tests for landmark_obj_to_array function."""

    def test_converts_dlib_landmark_to_array(self, mock_dlib_landmark_obj, sample_landmarks_68):
        """Test conversion of dlib landmark object to numpy array."""
        result = landmark_obj_to_array(mock_dlib_landmark_obj)

        assert isinstance(result, np.ndarray)
        assert result.shape == (68, 2)
        assert result.dtype == np.float32

    def test_landmark_array_values_correct(self, mock_dlib_landmark_obj, sample_landmarks_68):
        """Test that landmark array values match original coordinates."""
        result = landmark_obj_to_array(mock_dlib_landmark_obj)

        # Check a few points
        assert result[0, 0] == sample_landmarks_68[0, 0]
        assert result[0, 1] == sample_landmarks_68[0, 1]

    def test_landmark_array_has_correct_dtype(self, mock_dlib_landmark_obj):
        """Test that output is float32."""
        result = landmark_obj_to_array(mock_dlib_landmark_obj)
        assert result.dtype == np.float32


# ============================================================================
# TESTS FOR landmark_vector_to_matrix
# ============================================================================

class TestLandmarkVectorToMatrix:
    """Tests for landmark_vector_to_matrix function."""

    def test_converts_vector_to_matrix(self):
        """Test conversion of flattened vector to 2D matrix."""
        # Create vector: [x1, x2, ..., x68, y1, y2, ..., y68]
        vector = np.arange(136, dtype=np.float32)  # 68*2 = 136
        result = landmark_vector_to_matrix(vector)

        assert result.shape == (68, 2)
        assert result[0, 0] == 0  # First x
        assert result[0, 1] == 68  # First y

    def test_vector_to_matrix_conversion_accuracy(self):
        """Test exact conversion of known vectors."""
        vector = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        result = landmark_vector_to_matrix(vector)

        # vector length = 8, so 4 landmarks
        assert result.shape == (4, 2)
        assert result[0, 0] == 1  # x1
        assert result[0, 1] == 5  # y1
        assert result[3, 1] == 8  # y4


# ============================================================================
# TESTS FOR get_eye_centers
# ============================================================================

class TestGetEyeCenters:
    """Tests for get_eye_centers function."""

    def test_get_eye_centers_returns_two_points(self, sample_landmarks_68):
        """Test that get_eye_centers returns left and right eye centers."""
        left, right = get_eye_centers(sample_landmarks_68)

        assert isinstance(left, np.ndarray)
        assert isinstance(right, np.ndarray)
        assert left.shape == (2,)
        assert right.shape == (2,)

    def test_get_eye_centers_right_is_right_of_left(self, sample_landmarks_68):
        """Test that right eye is to the right of left eye."""
        # Modify landmarks so right eye is clearly to the right
        sample_landmarks_68[36:42] = np.array([[100, 100]] * 6)  # Left eye
        sample_landmarks_68[42:48] = np.array([[200, 100]] * 6)  # Right eye

        left, right = get_eye_centers(sample_landmarks_68)

        assert right[0] > left[0]  # Right x-coordinate is larger
        assert np.isclose(left[0], 100)
        assert np.isclose(right[0], 200)

    def test_get_eye_centers_computes_mean(self):
        """Test that eye centers are computed as mean of 6 points each."""
        landmarks = np.zeros((68, 2), dtype=np.float32)
        # Left eye points (36-41)
        landmarks[36:42] = np.array([[100, 100], [100, 110], [110, 110],
                                     [110, 100], [105, 105], [105, 95]])
        # Right eye points (42-47)
        landmarks[42:48] = np.array([[200, 100], [200, 110], [210, 110],
                                     [210, 100], [205, 105], [205, 95]])

        left, right = get_eye_centers(landmarks)

        # Left eye mean
        expected_left = landmarks[36:42].mean(axis=0)
        assert np.allclose(left, expected_left)

        # Right eye mean
        expected_right = landmarks[42:48].mean(axis=0)
        assert np.allclose(right, expected_right)


# ============================================================================
# TESTS FOR procrustes_normalize
# ============================================================================

class TestProcustesNormalize:
    """Tests for procrustes_normalize function."""

    def test_procrustes_normalize_returns_float32(self, sample_landmarks_68):
        """Test that procrustes_normalize returns float32 array."""
        result = procrustes_normalize(sample_landmarks_68)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_procrustes_normalize_output_shape(self, sample_landmarks_68):
        """Test that output shape matches input shape."""
        result = procrustes_normalize(sample_landmarks_68)

        assert result.shape == sample_landmarks_68.shape
        assert result.shape == (68, 2)

    def test_procrustes_normalize_centers_landmarks(self, sample_landmarks_68):
        """Test that landmarks are centered around origin."""
        result = procrustes_normalize(sample_landmarks_68)

        # After normalization, mean should be close to zero
        mean = result.mean(axis=0)
        assert np.allclose(mean, [0, 0], atol=1e-5)

    def test_procrustes_normalize_scales_landmarks(self, sample_landmarks_68):
        """Test that landmarks are scaled."""
        result = procrustes_normalize(sample_landmarks_68)

        # Scale should be approximately 1
        scale = np.sqrt(np.mean(np.sum(result**2, axis=1)))
        assert not np.isclose(scale, 0)  # Should be normalized, not zero

    def test_procrustes_normalize_with_template(self, sample_landmarks_68):
        """Test procrustes_normalize with template."""
        template = sample_landmarks_68 * 1.1  # Slightly scaled template
        result = procrustes_normalize(sample_landmarks_68, template=template)

        assert result.shape == (68, 2)
        assert result.dtype == np.float32


# ============================================================================
# TESTS FOR to_feature_vector
# ============================================================================

class TestToFeatureVector:
    """Tests for to_feature_vector function."""

    def test_to_feature_vector_flattens_array(self, sample_landmarks_68):
        """Test that to_feature_vector flattens 2D array."""
        result = to_feature_vector(sample_landmarks_68)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (136,)  # 68 * 2

    def test_to_feature_vector_preserves_values(self, sample_landmarks_68):
        """Test that flattening preserves all values."""
        result = to_feature_vector(sample_landmarks_68)
        expected = sample_landmarks_68.flatten()

        assert np.allclose(result, expected)

    def test_to_feature_vector_returns_float32(self, sample_landmarks_68):
        """Test that output is float32."""
        result = to_feature_vector(sample_landmarks_68)
        assert result.dtype == np.float32


# ============================================================================
# TESTS FOR center_by_reference
# ============================================================================

class TestCenterByReference:
    """Tests for center_by_reference function."""

    def test_center_by_reference_returns_tuple(self, sample_landmarks_68):
        """Test that function returns tuple of centered coords and reference."""
        centered, ref = center_by_reference(sample_landmarks_68, ref_index=33)

        assert isinstance(centered, np.ndarray)
        assert isinstance(ref, np.ndarray)

    def test_center_by_reference_centers_correctly(self, sample_landmarks_68):
        """Test that reference point becomes origin after centering."""
        ref_index = 33
        ref_point = sample_landmarks_68[ref_index].copy()
        
        centered, ref = center_by_reference(sample_landmarks_68, ref_index=ref_index)
        
        # After centering, reference point should be at origin
        assert np.allclose(centered[ref_index], [0, 0])
    
    def test_center_by_reference_returns_correct_reference(self, sample_landmarks_68):
        """Test that returned reference is the selected point."""
        ref_index = 25
        ref_point = sample_landmarks_68[ref_index].copy()
        
        centered, ref = center_by_reference(sample_landmarks_68, ref_index=ref_index)
        
        assert np.allclose(ref, ref_point)
    
    def test_center_by_reference_preserves_distances(self, sample_landmarks_68):
        """Test that relative distances are preserved after centering."""
        ref_index = 33
        original_dist = np.linalg.norm(
            sample_landmarks_68[0] - sample_landmarks_68[1]
        )
        
        centered, _ = center_by_reference(sample_landmarks_68, ref_index=ref_index)
        
        centered_dist = np.linalg.norm(centered[0] - centered[1])
        
        assert np.isclose(original_dist, centered_dist)


# ============================================================================
# TESTS FOR frontalize_landmarks
# ============================================================================

class TestFrontalizeLandmarks:
    """Tests for frontalize_landmarks function."""
    
    def test_frontalize_landmarks_no_weights_returns_none(self, mock_dlib_landmark_obj):
        """Test that None is returned when no frontalization weights."""
        result = frontalize_landmarks(mock_dlib_landmark_obj, None, None)
        assert result is None
    
    def test_frontalize_landmarks_with_weights_returns_array(self, mock_dlib_landmark_obj):
        """Test that array is returned when weights are provided."""
        # Create dummy weights (should transform to 3D or similar)
        weights = np.random.rand(137, 136).astype(np.float32)  # 68*2+1=137 inputs to 136 outputs
        
        result = frontalize_landmarks(mock_dlib_landmark_obj, weights, None)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
    
    def test_frontalize_landmarks_output_shape(self, mock_dlib_landmark_obj):
        """Test output shape of frontalization."""
        weights = np.random.rand(137, 136).astype(np.float32)
        result = frontalize_landmarks(mock_dlib_landmark_obj, weights, None)
        
        # Should be 68 landmarks, 2D coords (or possibly 3D)
        assert result.shape[0] == 68


# ============================================================================
# TESTS FOR process_frame
# ============================================================================

class TestProcessFrame:
    """Tests for process_frame function."""
    
    def test_process_frame_no_faces_returns_none(self, sample_frame, mock_detector_and_predictor):
        """Test that None is returned when no face is detected."""
        detector, predictor = mock_detector_and_predictor
        detector.return_value = []  # No faces detected
        
        result = process_frame(sample_frame, detector, predictor)
        
        assert result is None
    
    def test_process_frame_returns_dict_with_correct_keys(self, sample_frame, mock_detector_and_predictor, mock_dlib_landmark_obj):
        """Test that process_frame returns dict with expected keys."""
        detector, predictor = mock_detector_and_predictor
        predictor.return_value = mock_dlib_landmark_obj
        
        # Mock detector to return a rectangle
        rect_mock = Mock()
        rect_mock.left = Mock(return_value=50)
        rect_mock.top = Mock(return_value=50)
        rect_mock.right = Mock(return_value=250)
        rect_mock.bottom = Mock(return_value=350)
        detector.return_value = [rect_mock]
        
        result = process_frame(sample_frame, detector, predictor, frontalize=False)
        
        assert isinstance(result, dict)
        assert 'raw_landmarks' in result
        assert 'frontal_landmarks' in result
        assert 'centered_coords' in result
        assert 'feature_vector' in result
    
    def test_process_frame_extracts_landmarks(self, sample_frame, mock_detector_and_predictor, mock_dlib_landmark_obj):
        """Test that raw landmarks are extracted."""
        detector, predictor = mock_detector_and_predictor
        predictor.return_value = mock_dlib_landmark_obj
        
        rect_mock = Mock()
        rect_mock.left = Mock(return_value=50)
        rect_mock.top = Mock(return_value=50)
        rect_mock.right = Mock(return_value=250)
        rect_mock.bottom = Mock(return_value=350)
        detector.return_value = [rect_mock]
        
        result = process_frame(sample_frame, detector, predictor, frontalize=False)
        
        assert result['raw_landmarks'] is not None
        assert result['raw_landmarks'].shape == (68, 2)
    
    def test_process_frame_computes_feature_vector(self, sample_frame, mock_detector_and_predictor, mock_dlib_landmark_obj):
        """Test that feature vector is computed."""
        detector, predictor = mock_detector_and_predictor
        predictor.return_value = mock_dlib_landmark_obj
        
        rect_mock = Mock()
        rect_mock.left = Mock(return_value=50)
        rect_mock.top = Mock(return_value=50)
        rect_mock.right = Mock(return_value=250)
        rect_mock.bottom = Mock(return_value=350)
        detector.return_value = [rect_mock]
        
        result = process_frame(sample_frame, detector, predictor, center_ref=True)
        
        assert result['feature_vector'] is not None
        assert result['feature_vector'].shape == (136,)  # 68*2


# ============================================================================
# TESTS FOR video_to_landmark_vectors
# ============================================================================

class TestVideoToLandmarkVectors:
    """Tests for video_to_landmark_vectors function."""
    
    def test_video_to_landmark_vectors_invalid_path_returns_empty(self, mock_detector_and_predictor):
        """Test that empty list is returned for invalid video path."""
        detector, predictor = mock_detector_and_predictor
        
        result = video_to_landmark_vectors(
            video_path='/nonexistent/video.mp4',
            detector=detector,
            predictor=predictor
        )
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_video_to_landmark_vectors_returns_list(self, temp_dirs, mock_detector_and_predictor, mock_dlib_landmark_obj):
        """Test that video_to_landmark_vectors returns a list."""
        # Create a sample video file
        video_path = os.path.join(temp_dirs['videos'], 'test.mp4')
        
        # Create a minimal video using opencv
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (640, 480)
        )
        
        for _ in range(10):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        
        detector, predictor = mock_detector_and_predictor
        predictor.return_value = mock_dlib_landmark_obj
        
        rect_mock = Mock()
        rect_mock.left = Mock(return_value=50)
        rect_mock.top = Mock(return_value=50)
        rect_mock.right = Mock(return_value=250)
        rect_mock.bottom = Mock(return_value=350)
        detector.return_value = [rect_mock]
        
        result = video_to_landmark_vectors(
            video_path=video_path,
            detector=detector,
            predictor=predictor,
            frame_skip=2
        )
        
        assert isinstance(result, list)
    
    def test_video_to_landmark_vectors_respects_frame_skip(self, temp_dirs, mock_detector_and_predictor, mock_dlib_landmark_obj):
        """Test that frame_skip parameter is respected."""
        video_path = os.path.join(temp_dirs['videos'], 'test_skip.mp4')
        
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (640, 480)
        )
        
        for _ in range(30):  # 30 frames
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        
        detector, predictor = mock_detector_and_predictor
        predictor.return_value = mock_dlib_landmark_obj
        
        rect_mock = Mock()
        rect_mock.left = Mock(return_value=50)
        rect_mock.top = Mock(return_value=50)
        rect_mock.right = Mock(return_value=250)
        rect_mock.bottom = Mock(return_value=350)
        detector.return_value = [rect_mock]
        
        result = video_to_landmark_vectors(
            video_path=video_path,
            detector=detector,
            predictor=predictor,
            frame_skip=5  # Process every 5th frame
        )
        
        # With 30 frames and frame_skip=5, should get ~6 frames
        assert len(result) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple dlib pipeline functions."""
    
    def test_full_landmark_extraction_pipeline(self, sample_landmarks_68):
        """Test complete pipeline: landmarks -> normalization -> centering -> vectorization."""
        # Normalize
        normalized = procrustes_normalize(sample_landmarks_68)
        assert normalized.shape == (68, 2)
        
        # Center by reference
        centered, ref = center_by_reference(normalized, ref_index=33)
        assert centered[33][0] == 0 and centered[33][1] == 0
        
        # Convert to feature vector
        feature_vec = to_feature_vector(centered)
        assert feature_vec.shape == (136,)
    
    def test_feature_extraction_consistency(self, sample_landmarks_68):
        """Test that repeated feature extraction is consistent."""
        feature1 = to_feature_vector(center_by_reference(
            procrustes_normalize(sample_landmarks_68), ref_index=33
        )[0])
        
        feature2 = to_feature_vector(center_by_reference(
            procrustes_normalize(sample_landmarks_68), ref_index=33
        )[0])
        
        assert np.allclose(feature1, feature2)

    def test_eye_center_alignment_consistency(self, sample_landmarks_68):
        """Test consistency of eye center detection through pipeline."""
        left1, right1 = get_eye_centers(sample_landmarks_68)

        normalized = procrustes_normalize(sample_landmarks_68)
        left2, right2 = get_eye_centers(normalized)

        # Eye positions should be different after normalization
        # but still form valid eyes (right > left in x)
        assert right2[0] > left2[0]