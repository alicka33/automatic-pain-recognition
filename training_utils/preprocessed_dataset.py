import os
import pandas as pd 
from pathlib import Path
from typing import Optional, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    Dataset for processed per-video .npy files.

    Parameters
    ----------
    dataset_name : str
        'train' / 'val' / 'test' - used to find folder and metadata CSV:
        CSV expected at: os.path.join(processed_data_dir, f'{dataset_name}_processed_metadata.csv')
        and npy files inside os.path.join(processed_data_dir, dataset_name).
        CSV must contain columns: 'npy_path' (relative or basename) and 'label'.
    processed_data_dir : str | Path
        Root folder where processed outputs are stored.
    indices : array-like or None
        If provided, selects a subset of features (columns) AFTER Euclidean reduction
        (e.g. top-100 indices). Provide as numpy array or list of ints.
    compute_euclidean : bool
        If True expects raw 3D coords per frame (shape T x (N_points*3)) and computes Euclidean
        distances from center_point_index → resulting vector size N_points.
        If False assumes .npy already contains feature vectors (T x F) and uses them directly.
    center_point_index : int
        Index of the central landmark (0..N_points-1) used for distance calculation.
        Default matches many landmark schemes (33).
    num_coords_per_point : int
        Usually 3 (X,Y,Z).
    max_sequence_length : int
        Pad/truncate sequences to this length.
    selected_labels : iterable or None
        If provided, only rows whose label is in selected_labels will be included.
    label_map : dict or None
        If provided, remap labels using label_map[label].
    pad_value : float
        Value used to pad shorter sequences.
    """

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

        # NPY folder
        self.npy_folder = self.processed_data_dir / self.dataset_name
        if not self.npy_folder.exists():
            raise FileNotFoundError(f"NPY folder not found: {self.npy_folder}")

        if len(self.df) == 0:
            raise ValueError(f"No samples found for dataset '{dataset_name}' (after filtering).")

    def __len__(self):
        return len(self.df)

    def _load_npy(self, npy_path: Path) -> np.ndarray:
        if not npy_path.exists():
            raise FileNotFoundError(f"NPY file missing: {npy_path}")
        arr = np.load(str(npy_path)).astype(np.float32)
        return arr

    def _compute_euclidean(self, seq_3d: np.ndarray) -> np.ndarray:
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
        # features shape: (T, F)
        if self.indices is not None:
            if max(self.indices) >= features.shape[1]:
                raise IndexError("Provided indices contain index >= feature dimension.")
            return features[:, self.indices]
        return features

    def _pad_or_truncate(self, arr: np.ndarray) -> np.ndarray:
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
        row = self.df.iloc[idx]
        npy_path_str = row['npy_path']

        # If stored path is relative, try to resolve in npy_folder; otherwise use absolute
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

        # Label handling
        label_raw = int(row['label'])
        if self.label_map is not None:
            label = int(self.label_map.get(label_raw, label_raw))
        else:
            label = label_raw
        label_tensor = torch.tensor(label, dtype=torch.long)

        return sequence_tensor, label_tensor
