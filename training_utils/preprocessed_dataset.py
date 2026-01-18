import os
import pandas as pd 
from pathlib import Path
from typing import Optional, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """Load .npy sequences from processed_data_dir/{dataset_name}/ with optional Euclidean distances."""

    def __init__(
        self,
        dataset_name: str,
        processed_data_dir: str,
        indices: Optional[Sequence[int]] = None,
        compute_euclidean: bool = True,
        center_point_index: int = 2,
        num_coords_per_point: int = 3,
        max_sequence_length: int = 46,
        selected_labels: Optional[Iterable[int]] = None,
        label_map: Optional[dict] = None,
        pad_value: float = 0.0,
    ):
        self.dataset_name = dataset_name
        self.processed_data_dir = Path(processed_data_dir)
        self.indices = np.array(indices, dtype=int) if indices is not None else None
        self.compute_euclidean = compute_euclidean
        self.center_point_index = int(center_point_index)
        self.num_coords_per_point = int(num_coords_per_point)
        self.max_sequence_length = int(max_sequence_length)
        self.selected_labels = set(selected_labels) if selected_labels is not None else None
        self.label_map = label_map
        self.pad_value = float(pad_value)

        csv_path = self.processed_data_dir / f"{self.dataset_name}_processed_metadata.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Optional filter by labels (before building indices)
        if self.selected_labels is not None:
            self.df = self.df[self.df['label'].isin(self.selected_labels)].reset_index(drop=True)

        self.npy_folder = self.processed_data_dir / self.dataset_name
        if not self.npy_folder.exists():
            raise FileNotFoundError(f"NPY folder not found: {self.npy_folder}")

        if len(self.df) == 0:
            raise ValueError(f"No samples found for dataset '{dataset_name}' (after filtering).")

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.df)

    def _load_npy(self, npy_path: Path) -> np.ndarray:
        """Load .npy file and return as float32 array."""
        if not npy_path.exists():
            raise FileNotFoundError(f"NPY file missing: {npy_path}")
        arr = np.load(str(npy_path)).astype(np.float32)
        return arr

    def _compute_euclidean(self, seq_3d: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances from center landmark to all landmarks."""
        # seq_3d shape: (T, N_points * num_coords_per_point)
        T, F_full = seq_3d.shape
        if F_full % self.num_coords_per_point != 0:
            raise ValueError(f"Invalid shape: F_full ({F_full}) not divisible by num_coords_per_point ({self.num_coords_per_point})")
        N_points = F_full // self.num_coords_per_point
        points = seq_3d.reshape(T, N_points, self.num_coords_per_point)
        if not (0 <= self.center_point_index < N_points):
            raise IndexError(f"center_point_index {self.center_point_index} out of range (0..{N_points-1})")
        center = points[:, self.center_point_index, :]  # (T, 3)
        center_rep = center[:, None, :]  # (T,1,3)
        diff = points - center_rep  # (T, N_points, 3)
        dists = np.linalg.norm(diff, axis=2)  # (T, N_points)
        return dists

    def _select_features(self, features: np.ndarray) -> np.ndarray:
        """Select subset of features using stored indices."""
        # features shape: (T, F)
        if self.indices is not None:
            if max(self.indices) >= features.shape[1]:
                raise IndexError("Provided indices contain index >= feature dimension.")
            return features[:, self.indices]
        return features

    def _pad_or_truncate(self, arr: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to max_sequence_length."""
        # arr shape (T, F)
        T, F = arr.shape
        if T == self.max_sequence_length:
            return arr
        if T < self.max_sequence_length:
            pad_shape = (self.max_sequence_length - T, F)
            pad = np.full(pad_shape, self.pad_value, dtype=arr.dtype)
            return np.vstack([arr, pad])
        return arr[: self.max_sequence_length]

    def __getitem__(self, idx):
        """Load sequence and label, apply transformations, return as tensors."""
        row = self.df.iloc[idx]
        npy_path_str = row['npy_path']

        npy_filename = Path(npy_path_str).name
        npy_full = self.npy_folder / npy_filename

        seq = self._load_npy(npy_full)   # (T, F_full) or (T, F)

        if self.compute_euclidean:
            features = self._compute_euclidean(seq)  # (T, N_points)
        else:
            features = seq  # assume already (T, F)

        features = self._select_features(features)  # (T, num_features)
        features = self._pad_or_truncate(features)  # (max_seq_len, num_features)

        sequence_tensor = torch.from_numpy(features.astype(np.float32))  # (T_max, F)

        label_raw = int(row['label'])
        if self.label_map is not None:
            label = int(self.label_map.get(label_raw, label_raw))
        else:
            label = label_raw
        label_tensor = torch.tensor(label, dtype=torch.long)

        return sequence_tensor, label_tensor
