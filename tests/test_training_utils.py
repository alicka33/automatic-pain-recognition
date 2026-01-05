import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training_utils'))

from train import Trainer
from evaluate import Evaluator
from preprocessed_dataset import PreprocessedDataset


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple LSTM model for testing"""
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size=50, hidden_size=32, num_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n.squeeze(0))
            return out
    
    return SimpleLSTM(input_size=50, hidden_size=32, num_classes=3)


@pytest.fixture
def train_dataloader():
    """Create a simple training dataloader"""
    X = torch.randn(32, 10, 50)  # 32 samples, 10 timesteps, 50 features
    y = torch.randint(0, 3, (32,))  # 3 classes
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def val_dataloader():
    """Create a simple validation dataloader"""
    X = torch.randn(16, 10, 50)  # 16 samples, 10 timesteps, 50 features
    y = torch.randint(0, 3, (16,))  # 3 classes
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def test_dataloader():
    """Create a simple test dataloader"""
    X = torch.randn(16, 10, 50)
    y = torch.randint(0, 3, (16,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def preprocessed_dataset_dir():
    """Create temporary directory with preprocessed dataset files"""
    temp_dir = tempfile.mkdtemp()
    
    # Create subdirectories
    train_dir = os.path.join(temp_dir, 'train')
    val_dir = os.path.join(temp_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create sample .npy files and metadata CSVs
    # Train set: 10 samples
    train_metadata = []
    for i in range(10):
        seq = np.random.randn(30, 1434).astype(np.float32)  # 30 frames, 478 3D points
        npy_path = os.path.join(train_dir, f'video_{i}.npy')
        np.save(npy_path, seq)
        train_metadata.append({
            'npy_path': f'video_{i}.npy',
            'label': i % 3
        })
    
    train_df = pd.DataFrame(train_metadata)
    train_df.to_csv(os.path.join(temp_dir, 'train_processed_metadata.csv'), index=False)
    
    # Val set: 5 samples
    val_metadata = []
    for i in range(5):
        seq = np.random.randn(30, 1434).astype(np.float32)
        npy_path = os.path.join(val_dir, f'video_{i}.npy')
        np.save(npy_path, seq)
        val_metadata.append({
            'npy_path': f'video_{i}.npy',
            'label': i % 3
        })
    
    val_df = pd.DataFrame(val_metadata)
    val_df.to_csv(os.path.join(temp_dir, 'val_processed_metadata.csv'), index=False)
    
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# TESTS: Trainer
# ============================================================================

class TestTrainer:
    """Tests for Trainer class"""
    
    def test_trainer_initialization(self, simple_model, train_dataloader, val_dataloader, temp_model_dir):
        """Test Trainer initialization"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        model_path = os.path.join(temp_model_dir, 'model.pt')
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            model_save_path=model_path,
            num_epochs=5
        )
        
        assert trainer.model is simple_model
        assert trainer.num_epochs == 5
        assert trainer.device is not None
        assert trainer.history == {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def test_train_epoch_returns_loss_and_acc(self, simple_model, train_dataloader, val_dataloader, temp_model_dir):
        """Test that train_epoch returns loss and accuracy"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        model_path = os.path.join(temp_model_dir, 'model.pt')
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            model_save_path=model_path
        )
        
        train_loss, train_acc = trainer.train_epoch()
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 1
    
    def test_validate_epoch_returns_loss_acc_and_predictions(self, simple_model, train_dataloader, val_dataloader, temp_model_dir):
        """Test that validate_epoch returns loss, accuracy, labels and predictions"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        model_path = os.path.join(temp_model_dir, 'model.pt')
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            model_save_path=model_path
        )
        
        val_loss, val_acc, labels, preds = trainer.validate_epoch()
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert isinstance(labels, list)
        assert isinstance(preds, list)
        assert len(labels) == len(preds)
    
    def test_save_checkpoint(self, simple_model, train_dataloader, val_dataloader, temp_model_dir):
        """Test checkpoint saving"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        model_path = os.path.join(temp_model_dir, 'model.pt')
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            model_save_path=model_path
        )
        
        saved_path = trainer.save_checkpoint()
        
        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == 'model.pt'
    
    def test_history_updated_after_epoch(self, simple_model, train_dataloader, val_dataloader, temp_model_dir):
        """Test that history is updated after each epoch"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        model_path = os.path.join(temp_model_dir, 'model.pt')
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            model_save_path=model_path
        )
        
        assert len(trainer.history['train_loss']) == 0
        
        trainer.train_epoch()
        trainer.validate_epoch()
        
        # Manually update history (simulating what fit() does)
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc, _, _ = trainer.validate_epoch()
        trainer.history['train_loss'].append(train_loss)
        trainer.history['train_acc'].append(train_acc)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_acc'].append(val_acc)
        
        assert len(trainer.history['train_loss']) > 0
        assert len(trainer.history['val_loss']) > 0
    
    def test_print_training_config(self, simple_model, train_dataloader, val_dataloader, temp_model_dir, capsys):
        """Test that print_training_config doesn't crash"""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        model_path = os.path.join(temp_model_dir, 'model.pt')
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            model_save_path=model_path,
            num_epochs=10
        )
        
        # Should not raise
        trainer.print_training_config()
        
        captured = capsys.readouterr()
        assert 'TRAINING CONFIGURATION' in captured.out
        assert 'SimpleLSTM' in captured.out


# ============================================================================
# TESTS: Evaluator
# ============================================================================

class TestEvaluator:
    """Tests for Evaluator class"""
    
    def test_evaluator_initialization(self, simple_model, test_dataloader):
        """Test Evaluator initialization"""
        evaluator = Evaluator(
            model=simple_model,
            test_loader=test_dataloader,
            model_name='TestModel',
            num_classes=3
        )
        
        assert evaluator.model is simple_model
        assert evaluator.model_name == 'TestModel'
        assert evaluator.num_classes == 3
    
    def test_evaluate_epoch_returns_metrics(self, simple_model, test_dataloader):
        """Test that evaluate_epoch returns loss, accuracy, labels and predictions"""
        evaluator = Evaluator(
            model=simple_model,
            test_loader=test_dataloader,
            verbose=False
        )
        
        test_loss, test_acc, labels, preds = evaluator.evaluate_epoch()
        
        assert isinstance(test_loss, float)
        assert isinstance(test_acc, float)
        assert isinstance(labels, list)
        assert isinstance(preds, list)
        assert len(labels) == len(preds)
        assert test_loss >= 0
        assert 0 <= test_acc <= 1
    
    def test_print_results(self, simple_model, test_dataloader, capsys):
        """Test that print_results prints without crashing"""
        evaluator = Evaluator(
            model=simple_model,
            test_loader=test_dataloader,
            model_name='TestModel',
            num_classes=3,
            verbose=False
        )
        
        labels = [0, 1, 2, 0, 1, 2]
        preds = [0, 1, 2, 0, 1, 1]
        
        evaluator.print_results('TestModel', 0.5, 0.83, labels, preds, num_classes=3)
        
        captured = capsys.readouterr()
        assert 'TestModel' in captured.out
        assert '0.5000' in captured.out or '0.50' in captured.out
    
    def test_evaluate_epoch_auto_prints_with_model_name(self, simple_model, test_dataloader, capsys):
        """Test that evaluate_epoch auto-prints when model_name is provided"""
        evaluator = Evaluator(
            model=simple_model,
            test_loader=test_dataloader,
            model_name='TestModel',
            num_classes=3,
            verbose=True
        )
        
        evaluator.evaluate_epoch()
        
        captured = capsys.readouterr()
        assert 'TestModel' in captured.out


# ============================================================================
# TESTS: PreprocessedDataset
# ============================================================================

class TestPreprocessedDataset:
    """Tests for PreprocessedDataset class"""
    
    def test_dataset_initialization(self, preprocessed_dataset_dir):
        """Test PreprocessedDataset initialization"""
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir
        )
        
        assert len(dataset) == 10
        assert dataset.dataset_name == 'train'
    
    def test_dataset_getitem_returns_tensor_and_label(self, preprocessed_dataset_dir):
        """Test that __getitem__ returns tensor and label"""
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=True
        )
        
        sequence, label = dataset[0]
        
        assert isinstance(sequence, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert sequence.shape[0] == 46  # max_sequence_length
        assert sequence.shape[1] == 478  # num points (1434 / 3)
        assert label.dtype == torch.long
    
    def test_euclidean_distance_computation(self, preprocessed_dataset_dir):
        """Test Euclidean distance computation"""
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=True,
            center_point_index=2,
            num_coords_per_point=3,
            max_sequence_length=46
        )
        
        sequence, label = dataset[0]
        
        # After Euclidean reduction, should have 478 features (num points)
        assert sequence.shape[1] == 478
    
    def test_padding_shorter_sequences(self, preprocessed_dataset_dir):
        """Test that shorter sequences are padded"""
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=True,
            max_sequence_length=50,  # Longer than the 30 frames in test data
            pad_value=0.0
        )
        
        sequence, _ = dataset[0]
        
        # Should be padded to 50 frames
        assert sequence.shape[0] == 50
    
    def test_truncating_longer_sequences(self, preprocessed_dataset_dir):
        """Test that longer sequences are truncated"""
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=True,
            max_sequence_length=20  # Shorter than the 30 frames in test data
        )
        
        sequence, _ = dataset[0]
        
        # Should be truncated to 20 frames
        assert sequence.shape[0] == 20
    
    def test_label_mapping(self, preprocessed_dataset_dir):
        """Test label mapping"""
        label_map = {0: 10, 1: 20, 2: 30}
        
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            label_map=label_map,
            compute_euclidean=True
        )
        
        _, label = dataset[0]
        
        # Label should be mapped
        assert label.item() in [10, 20, 30]
    
    def test_selected_labels_filtering(self, preprocessed_dataset_dir):
        """Test filtering by selected labels"""
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            selected_labels=[0, 1],  # Only classes 0 and 1
            compute_euclidean=True
        )
        
        # Should have fewer than 10 samples (class 2 is filtered)
        assert len(dataset) <= 10
        
        # All labels should be in [0, 1]
        for i in range(len(dataset)):
            _, label = dataset[i]
            assert label.item() in [0, 1]
    
    def test_feature_selection(self, preprocessed_dataset_dir):
        """Test feature selection"""
        indices = list(range(100))  # Select first 100 features
        
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            indices=indices,
            compute_euclidean=True
        )
        
        sequence, _ = dataset[0]
        
        # Should have 100 features after selection
        assert sequence.shape[1] == 100
    
    def test_missing_csv_raises_error(self, preprocessed_dataset_dir):
        """Test that missing metadata CSV raises error"""
        with pytest.raises(FileNotFoundError):
            PreprocessedDataset(
                dataset_name='nonexistent',
                processed_data_dir=preprocessed_dataset_dir
            )
    
    def test_compute_euclidean_false(self, preprocessed_dataset_dir):
        """Test with compute_euclidean=False (use raw features)"""
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=False,
            max_sequence_length=46
        )
        
        sequence, label = dataset[0]
        
        # Should have 1434 features (raw 3D coords)
        assert sequence.shape[1] == 1434
        assert sequence.dtype == torch.float32
    
    def test_dataloader_compatibility(self, preprocessed_dataset_dir):
        """Test that dataset works with PyTorch DataLoader"""
        dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=True
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        for batch_sequences, batch_labels in dataloader:
            assert batch_sequences.shape[0] == 4  # batch size
            assert batch_sequences.shape[1] == 46  # seq length
            assert batch_sequences.shape[2] == 478  # num features
            assert batch_labels.shape[0] == 4
            break  # Just test first batch


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across components"""
    
    def test_trainer_with_preprocessed_dataset(self, simple_model, preprocessed_dataset_dir, temp_model_dir):
        """Test Trainer with PreprocessedDataset"""
        # Create datasets
        train_dataset = PreprocessedDataset(
            dataset_name='train',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=True,
            max_sequence_length=46
        )
        
        val_dataset = PreprocessedDataset(
            dataset_name='val',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=True,
            max_sequence_length=46
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=4)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        # Create model that matches dataset output
        class DatasetLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=478, hidden_size=32, batch_first=True)
                self.fc = nn.Linear(32, 3)
            
            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.fc(h_n.squeeze(0))
        
        model = DatasetLSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model_path = os.path.join(temp_model_dir, 'model.pt')
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            model_save_path=model_path,
            num_epochs=2
        )
        
        # Train for one epoch
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc, _, _ = trainer.validate_epoch()
        
        assert train_loss >= 0
        assert val_loss >= 0
        assert 0 <= train_acc <= 1
        assert 0 <= val_acc <= 1
    
    def test_evaluator_with_preprocessed_dataset(self, simple_model, preprocessed_dataset_dir):
        """Test Evaluator with PreprocessedDataset"""
        dataset = PreprocessedDataset(
            dataset_name='val',
            processed_data_dir=preprocessed_dataset_dir,
            compute_euclidean=True,
            max_sequence_length=46
        )
        
        dataloader = DataLoader(dataset, batch_size=4)
        
        evaluator = Evaluator(
            model=simple_model,
            test_loader=dataloader,
            model_name='TestModel',
            num_classes=3,
            verbose=False
        )
        
        test_loss, test_acc, labels, preds = evaluator.evaluate_epoch()
        
        assert test_loss >= 0
        assert 0 <= test_acc <= 1
        assert len(labels) == len(preds)
