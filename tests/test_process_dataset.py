import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_preparation'))

from data_preparation.process_dataset import (
    ProcessingConfig,
    ensure_dir,
    copy_to_local_cache,
    pad_or_truncate_sequence,
    process_single_video_row,
    process_dataframe_to_npy
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing"""
    temp_base = tempfile.mkdtemp()
    colab_root = os.path.join(temp_base, 'colab_root')
    local_cache = os.path.join(temp_base, 'cache')
    processed_dir = os.path.join(colab_root, 'data', 'processed')
    
    Path(colab_root).mkdir(parents=True, exist_ok=True)
    Path(local_cache).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    yield {
        'base': temp_base,
        'colab_root': colab_root,
        'cache': local_cache,
        'processed': processed_dir
    }
    
    # Cleanup
    shutil.rmtree(temp_base, ignore_errors=True)


@pytest.fixture
def test_config(temp_dirs):
    """Create a test ProcessingConfig"""
    return ProcessingConfig(
        colab_root=temp_dirs['colab_root'],
        dataset_subdir='data/BioVid_HeatPain/',
        processed_subdir='data/processed/',
        local_cache_dir=temp_dirs['cache'],
        max_sequence_length=46,
        feature_dim=1434
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with video metadata"""
    return pd.DataFrame({
        'video_path': ['video1.mp4', 'video2.avi', 'video3.mp4'],
        'label': [0, 1, 2]
    })


@pytest.fixture
def sample_sequence():
    """Create a sample feature sequence"""
    return np.random.randn(30, 1434).astype(np.float32)


@pytest.fixture
def dummy_video_processor():
    """Create a dummy video processor function"""
    def processor(video_path, frame_skip=3, visualize=False):
        # Return 25 frames of 1434-dimensional features
        return [np.random.randn(1434).astype(np.float32) for _ in range(25)]
    
    return processor


# ============================================================================
# TESTS: ProcessingConfig
# ============================================================================

class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass"""
    
    def test_default_initialization(self):
        """Test ProcessingConfig creates with default values"""
        config = ProcessingConfig()
        assert config.colab_root == '/content/drive/MyDrive/PainRecognitionProject/'
        assert config.max_sequence_length == 46
        assert config.feature_dim == 1434
    
    def test_custom_initialization(self):
        """Test ProcessingConfig with custom values"""
        config = ProcessingConfig(
            colab_root='/custom/root',
            max_sequence_length=50,
            feature_dim=1000
        )
        assert config.colab_root == '/custom/root'
        assert config.max_sequence_length == 50
        assert config.feature_dim == 1000
    
    def test_data_dir_property(self, test_config):
        """Test data_dir property construction"""
        expected = os.path.join(
            test_config.colab_root,
            test_config.dataset_subdir
        )
        assert test_config.data_dir == expected
    
    def test_processed_data_dir_property(self, test_config):
        """Test processed_data_dir property construction"""
        expected = os.path.join(
            test_config.colab_root,
            test_config.processed_subdir
        )
        assert test_config.processed_data_dir == expected


# ============================================================================
# TESTS: ensure_dir
# ============================================================================

class TestEnsureDir:
    """Tests for ensure_dir function"""
    
    def test_create_single_directory(self, temp_dirs):
        """Test creating a single directory"""
        new_dir = os.path.join(temp_dirs['base'], 'new_dir')
        assert not os.path.exists(new_dir)
        
        ensure_dir(new_dir)
        
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)
    
    def test_create_nested_directories(self, temp_dirs):
        """Test creating nested directory structure"""
        nested_dir = os.path.join(
            temp_dirs['base'], 'a', 'b', 'c', 'd'
        )
        assert not os.path.exists(nested_dir)
        
        ensure_dir(nested_dir)
        
        assert os.path.exists(nested_dir)
        assert os.path.isdir(nested_dir)
    
    def test_existing_directory_no_error(self, temp_dirs):
        """Test that existing directory doesn't raise error"""
        existing_dir = temp_dirs['cache']
        assert os.path.exists(existing_dir)
        
        # Should not raise
        ensure_dir(existing_dir)
        
        assert os.path.exists(existing_dir)


# ============================================================================
# TESTS: copy_to_local_cache
# ============================================================================

class TestCopyToLocalCache:
    """Tests for copy_to_local_cache function"""
    
    def test_copy_file_to_cache(self, temp_dirs):
        """Test copying a file to local cache"""
        # Create source file
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'test_video.mp4')
        with open(src_file, 'w') as f:
            f.write('test content')
        
        # Copy to cache
        local_path = copy_to_local_cache(src_file, temp_dirs['cache'])
        
        assert os.path.exists(local_path)
        assert os.path.dirname(local_path) == temp_dirs['cache']
        with open(local_path, 'r') as f:
            assert f.read() == 'test content'
    
    def test_copy_creates_cache_dir(self, temp_dirs):
        """Test that copy_to_local_cache creates cache directory if needed"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video.mp4')
        with open(src_file, 'w') as f:
            f.write('content')
        
        new_cache = os.path.join(temp_dirs['base'], 'new_cache')
        assert not os.path.exists(new_cache)
        
        local_path = copy_to_local_cache(src_file, new_cache)
        
        assert os.path.exists(new_cache)
        assert os.path.exists(local_path)
    
    def test_copy_nonexistent_file_raises_error(self, temp_dirs):
        """Test that copying nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            copy_to_local_cache(
                '/nonexistent/file.mp4',
                temp_dirs['cache']
            )
    
    def test_copy_preserves_filename(self, temp_dirs):
        """Test that copied file gets correct name"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'my_video.mp4')
        with open(src_file, 'w') as f:
            f.write('content')
        
        local_path = copy_to_local_cache(src_file, temp_dirs['cache'])
        
        assert 'my_video.mp4' in os.path.basename(local_path)


# ============================================================================
# TESTS: pad_or_truncate_sequence
# ============================================================================

class TestPadOrTruncateSequence:
    """Tests for pad_or_truncate_sequence function"""
    
    def test_pad_short_sequence(self, sample_sequence):
        """Test padding a sequence shorter than max_length"""
        max_length = 50
        feature_dim = 1434
        
        result = pad_or_truncate_sequence(sample_sequence, max_length, feature_dim)
        
        assert result.shape == (max_length, feature_dim)
        assert np.allclose(result[:30], sample_sequence)
        assert np.allclose(result[30:], 0)
    
    def test_truncate_long_sequence(self, sample_sequence):
        """Test truncating a sequence longer than max_length"""
        long_sequence = np.random.randn(100, 1434).astype(np.float32)
        max_length = 50
        feature_dim = 1434
        
        result = pad_or_truncate_sequence(long_sequence, max_length, feature_dim)
        
        assert result.shape == (max_length, feature_dim)
        assert np.allclose(result, long_sequence[:50])
    
    def test_exact_length_sequence(self, sample_sequence):
        """Test sequence with exact max_length"""
        exact_sequence = np.random.randn(46, 1434).astype(np.float32)
        max_length = 46
        feature_dim = 1434
        
        result = pad_or_truncate_sequence(exact_sequence, max_length, feature_dim)
        
        assert result.shape == (max_length, feature_dim)
        assert np.allclose(result, exact_sequence)
    
    def test_empty_sequence_returns_zeros(self):
        """Test that empty sequence returns zero array"""
        empty = np.array([]).reshape(0, 1434).astype(np.float32)
        max_length = 46
        feature_dim = 1434
        
        result = pad_or_truncate_sequence(empty, max_length, feature_dim)
        
        assert result.shape == (max_length, feature_dim)
        assert np.allclose(result, 0)
    
    def test_none_sequence_returns_zeros(self):
        """Test that None sequence returns zero array"""
        max_length = 46
        feature_dim = 1434
        
        result = pad_or_truncate_sequence(None, max_length, feature_dim)
        
        assert result.shape == (max_length, feature_dim)
        assert np.allclose(result, 0)
    
    def test_feature_dim_mismatch_raises_error(self, sample_sequence):
        """Test that feature dimension mismatch raises ValueError"""
        max_length = 50
        wrong_feature_dim = 1000
        
        with pytest.raises(ValueError, match="Feature dim mismatch"):
            pad_or_truncate_sequence(sample_sequence, max_length, wrong_feature_dim)
    
    def test_output_dtype_preserved(self, sample_sequence):
        """Test that output dtype is preserved or converted to float32"""
        result = pad_or_truncate_sequence(sample_sequence, 46, 1434)
        
        assert result.dtype in [np.float32, np.float64]


# ============================================================================
# TESTS: process_single_video_row
# ============================================================================

class TestProcessSingleVideoRow:
    """Tests for process_single_video_row function"""
    
    def test_successful_processing(self, temp_dirs, test_config, dummy_video_processor):
        """Test successful processing of a video row"""
        # Create source video file
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video1.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy video')
        
        row = pd.Series({
            'video_path': 'video1.mp4',
            'label': 0
        })
        
        result = process_single_video_row(
            row=row,
            data_dir=src_dir,
            npy_output_dir=test_config.processed_data_dir,
            local_cache_dir=test_config.local_cache_dir,
            video_processor=dummy_video_processor,
            frame_skip=3,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        assert result is not None
        assert 'npy_path' in result
        assert 'label' in result
        assert result['label'] == 0
        assert result['npy_path'].endswith('video1.npy')
    
    def test_saves_npy_file(self, temp_dirs, test_config, dummy_video_processor):
        """Test that .npy file is actually saved"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video1.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        row = pd.Series({'video_path': 'video1.mp4', 'label': 1})
        
        result = process_single_video_row(
            row=row,
            data_dir=src_dir,
            npy_output_dir=test_config.processed_data_dir,
            local_cache_dir=test_config.local_cache_dir,
            video_processor=dummy_video_processor,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        npy_path = os.path.join(test_config.processed_data_dir, 'video1.npy')
        assert os.path.exists(npy_path)
        
        loaded = np.load(npy_path)
        assert loaded.shape == (46, 1434)
    
    def test_handles_avi_extension(self, temp_dirs, test_config, dummy_video_processor):
        """Test that .avi files are converted to .npy correctly"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video.avi')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        row = pd.Series({'video_path': 'video.avi', 'label': 2})
        
        result = process_single_video_row(
            row=row,
            data_dir=src_dir,
            npy_output_dir=test_config.processed_data_dir,
            local_cache_dir=test_config.local_cache_dir,
            video_processor=dummy_video_processor,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        assert result is not None
        assert result['npy_path'].endswith('video.npy')
        npy_path = os.path.join(test_config.processed_data_dir, 'video.npy')
        assert os.path.exists(npy_path)
    
    def test_handles_empty_frame_list(self, temp_dirs, test_config):
        """Test that empty frame list is handled (returns zero sequence)"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        def empty_processor(video_path, frame_skip=3, visualize=False):
            return []
        
        row = pd.Series({'video_path': 'video.mp4', 'label': 0})
        
        result = process_single_video_row(
            row=row,
            data_dir=src_dir,
            npy_output_dir=test_config.processed_data_dir,
            local_cache_dir=test_config.local_cache_dir,
            video_processor=empty_processor,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        assert result is not None
        npy_path = os.path.join(test_config.processed_data_dir, 'video.npy')
        loaded = np.load(npy_path)
        assert loaded.shape == (46, 1434)
        assert np.allclose(loaded, 0)
    
    def test_cleans_up_temp_file(self, temp_dirs, test_config, dummy_video_processor):
        """Test that temporary local cache file is deleted after processing"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        row = pd.Series({'video_path': 'video.mp4', 'label': 0})
        
        # Count files before
        cache_files_before = len(os.listdir(test_config.local_cache_dir))
        
        process_single_video_row(
            row=row,
            data_dir=src_dir,
            npy_output_dir=test_config.processed_data_dir,
            local_cache_dir=test_config.local_cache_dir,
            video_processor=dummy_video_processor,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        # Count files after (should be same since temp was deleted)
        cache_files_after = len(os.listdir(test_config.local_cache_dir))
        assert cache_files_after == cache_files_before
    
    def test_error_handling_returns_none(self, temp_dirs, test_config):
        """Test that processing errors return None"""
        def failing_processor(video_path, frame_skip=3, visualize=False):
            raise RuntimeError("Processing failed")
        
        row = pd.Series({'video_path': 'nonexistent.mp4', 'label': 0})
        
        result = process_single_video_row(
            row=row,
            data_dir='/nonexistent/path',
            npy_output_dir=test_config.processed_data_dir,
            local_cache_dir=test_config.local_cache_dir,
            video_processor=failing_processor,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        assert result is None
    
    def test_label_preserved_in_metadata(self, temp_dirs, test_config, dummy_video_processor):
        """Test that label is correctly preserved in output metadata"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        for test_label in [0, 1, 2, 999]:
            row = pd.Series({'video_path': 'video.mp4', 'label': test_label})
            
            result = process_single_video_row(
                row=row,
                data_dir=src_dir,
                npy_output_dir=test_config.processed_data_dir,
                local_cache_dir=test_config.local_cache_dir,
                video_processor=dummy_video_processor,
                max_sequence_length=46,
                feature_dim=1434
            )
            
            assert result['label'] == test_label


# ============================================================================
# TESTS: process_dataframe_to_npy
# ============================================================================

class TestProcessDataframeToNpy:
    """Tests for process_dataframe_to_npy function"""
    
    def test_process_multiple_videos(self, temp_dirs, test_config, dummy_video_processor):
        """Test processing multiple videos from DataFrame"""
        # Create source videos
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        for name in ['video1.mp4', 'video2.mp4', 'video3.avi']:
            src_file = os.path.join(src_dir, name)
            with open(src_file, 'w') as f:
                f.write('dummy')
        
        df = pd.DataFrame({
            'video_path': ['video1.mp4', 'video2.mp4', 'video3.avi'],
            'label': [0, 1, 2]
        })
        
        modified_config = ProcessingConfig(
            colab_root=test_config.colab_root,
            dataset_subdir='',
            processed_subdir='data/processed/',
            local_cache_dir=test_config.local_cache_dir,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        result_df = process_dataframe_to_npy(
            df=df,
            dataset_name='test_dataset',
            video_processor=dummy_video_processor,
            config=modified_config,
            frame_skip=3,
            progress=False
        )
        
        assert len(result_df) == 3
        assert list(result_df['label']) == [0, 1, 2]
    
    def test_saves_metadata_csv(self, temp_dirs, test_config, dummy_video_processor):
        """Test that metadata CSV is saved"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        df = pd.DataFrame({'video_path': ['video.mp4'], 'label': [1]})
        
        modified_config = ProcessingConfig(
            colab_root=test_config.colab_root,
            dataset_subdir='',
            processed_subdir='data/processed/',
            local_cache_dir=test_config.local_cache_dir,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        process_dataframe_to_npy(
            df=df,
            dataset_name='test_dataset',
            video_processor=dummy_video_processor,
            config=modified_config,
            progress=False
        )
        
        csv_path = os.path.join(modified_config.processed_data_dir, 'test_dataset_processed_metadata.csv')
        assert os.path.exists(csv_path)
        
        meta_df = pd.read_csv(csv_path)
        assert len(meta_df) == 1
        assert meta_df.iloc[0]['label'] == 1
    
    def test_skips_failed_videos(self, temp_dirs, test_config):
        """Test that failed videos are skipped without stopping process"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        
        # Create one working, one that will fail (doesn't exist)
        src_file = os.path.join(src_dir, 'video_ok.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        def dummy_processor(video_path, frame_skip=3, visualize=False):
            return [np.random.randn(1434).astype(np.float32) for _ in range(25)]
        
        df = pd.DataFrame({
            'video_path': ['video_ok.mp4', 'video_fail.mp4'],
            'label': [0, 1]
        })
        
        modified_config = ProcessingConfig(
            colab_root=test_config.colab_root,
            dataset_subdir='',
            processed_subdir='data/processed/',
            local_cache_dir=test_config.local_cache_dir,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        result_df = process_dataframe_to_npy(
            df=df,
            dataset_name='test_dataset',
            video_processor=dummy_processor,
            config=modified_config,
            progress=False
        )
        
        # Only one video should be in result
        assert len(result_df) == 1
        assert result_df.iloc[0]['label'] == 0
    
    def test_uses_config_parameters(self, temp_dirs, dummy_video_processor):
        """Test that config parameters are used correctly"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        custom_config = ProcessingConfig(
            colab_root=temp_dirs['colab_root'],
            dataset_subdir='',
            processed_subdir='custom_processed/',
            local_cache_dir=temp_dirs['cache'],
            max_sequence_length=100,
            feature_dim=500
        )
        
        df = pd.DataFrame({'video_path': ['video.mp4'], 'label': [0]})
        
        def dummy_processor(video_path, frame_skip=3, visualize=False):
            return [np.random.randn(500).astype(np.float32) for _ in range(50)]
        
        result_df = process_dataframe_to_npy(
            df=df,
            dataset_name='test',
            video_processor=dummy_processor,
            config=custom_config,
            progress=False
        )
        
        # Check that npy file has correct shape based on config
        npy_path = os.path.join(custom_config.processed_data_dir, 'video.npy')
        loaded = np.load(npy_path)
        assert loaded.shape == (100, 500)
    
    def test_progress_bar_toggle(self, temp_dirs, test_config, dummy_video_processor):
        """Test that progress parameter works"""
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        src_file = os.path.join(src_dir, 'video.mp4')
        with open(src_file, 'w') as f:
            f.write('dummy')
        
        df = pd.DataFrame({'video_path': ['video.mp4'], 'label': [0]})
        
        modified_config = ProcessingConfig(
            colab_root=test_config.colab_root,
            dataset_subdir='',
            processed_subdir='data/processed/',
            local_cache_dir=test_config.local_cache_dir,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        # Should not raise error with progress=True
        process_dataframe_to_npy(
            df=df,
            dataset_name='test',
            video_processor=dummy_video_processor,
            config=modified_config,
            progress=True
        )
        
        # Should not raise error with progress=False
        process_dataframe_to_npy(
            df=df,
            dataset_name='test2',
            video_processor=dummy_video_processor,
            config=modified_config,
            progress=False
        )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across multiple functions"""
    
    def test_full_pipeline(self, temp_dirs, test_config, dummy_video_processor):
        """Test complete pipeline: DataFrame -> npy files -> metadata CSV"""
        # Setup
        src_dir = os.path.join(temp_dirs['base'], 'source')
        Path(src_dir).mkdir(parents=True)
        for i in range(3):
            src_file = os.path.join(src_dir, f'video_{i}.mp4')
            with open(src_file, 'w') as f:
                f.write(f'dummy video {i}')
        
        df = pd.DataFrame({
            'video_path': ['video_0.mp4', 'video_1.mp4', 'video_2.mp4'],
            'label': [0, 1, 0]
        })
        
        modified_config = ProcessingConfig(
            colab_root=test_config.colab_root,
            dataset_subdir='',
            processed_subdir='data/processed/',
            local_cache_dir=test_config.local_cache_dir,
            max_sequence_length=46,
            feature_dim=1434
        )
        
        # Process
        result_df = process_dataframe_to_npy(
            df=df,
            dataset_name='integration_test',
            video_processor=dummy_video_processor,
            config=modified_config,
            progress=False
        )
        
        # Verify results
        assert len(result_df) == 3
        
        # Check npy files
        for i in range(3):
            npy_path = os.path.join(modified_config.processed_data_dir, f'video_{i}.npy')
            assert os.path.exists(npy_path)
            loaded = np.load(npy_path)
            assert loaded.shape == (46, 1434)
        
        # Check metadata CSV
        csv_path = os.path.join(
            modified_config.processed_data_dir,
            'integration_test_processed_metadata.csv'
        )
        assert os.path.exists(csv_path)
        meta = pd.read_csv(csv_path)
        assert len(meta) == 3
        assert list(meta['label']) == [0, 1, 0]
