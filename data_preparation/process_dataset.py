import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
import os


@dataclass
class ProcessingConfig:
    colab_root: str = '/content/drive/MyDrive/PainRecognitionProject/'

    dataset_subdir: str = 'data/BioVid_HeatPain/'
    processed_subdir: str = 'data/BioVid_HeatPain_processed_478_xyz_frontalized/'
    reference_keypoints_path: str = 'data/key_points_xyz.npy'

    use_frontalization: bool = True
    local_cache_dir: str = '/content/temp_cache'

    max_sequence_length: int = 46
    feature_dim: int = 478 * 3  # default 1434

    @property
    def data_dir(self) -> str:
        return os.path.join(self.colab_root, self.dataset_subdir)

    @property
    def processed_data_dir(self) -> str:
        return os.path.join(self.colab_root, self.processed_subdir)

    @property
    def reference_path(self) -> str:
        return os.path.join(self.colab_root, self.reference_keypoints_path)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def copy_to_local_cache(drive_video_path: str, local_cache_dir: str) -> str:
    """
    Copy a file from drive path to local cache path and return local path.
    Raises FileNotFoundError if source doesn't exist.
    """
    ensure_dir(local_cache_dir)
    local_name = os.path.basename(drive_video_path).replace('/', '_')
    local_path = os.path.join(local_cache_dir, local_name)
    shutil.copyfile(drive_video_path, local_path)
    return local_path


def pad_or_truncate_sequence(sequence: np.ndarray, max_length: int, feature_dim: int) -> np.ndarray:
    """
    Return a sequence with shape (max_length, feature_dim) by padding with zeros or truncating.
    """
    if sequence is None or sequence.size == 0:
        return np.zeros((max_length, feature_dim), dtype=np.float32)

    current_len, cur_dim = sequence.shape
    if cur_dim != feature_dim:
        raise ValueError(f"Feature dim mismatch: expected {feature_dim}, got {cur_dim}")

    if current_len < max_length:
        pad_shape = (max_length - current_len, cur_dim)
        pad = np.zeros(pad_shape, dtype=sequence.dtype)
        return np.concatenate([sequence, pad], axis=0)
    elif current_len > max_length:
        return sequence[:max_length]
    else:
        return sequence


def process_single_video_row(
    row: pd.Series,
    data_dir: str,
    npy_output_dir: str,
    processed_data_root: str,
    local_cache_dir: str,
    video_processor: Callable[..., List[np.ndarray]],
    frame_skip: int = 3,
    max_sequence_length: int = 46,
    feature_dim: int = 1434,
    visualize: bool = False
) -> Optional[Dict]:
    """
    Process a single DataFrame row with keys 'video_path' and 'label'.
    Returns metadata dict {'npy_path': relative_path, 'label': label} or None on failure.
    """
    video_filename = row['video_path']
    label = row.get('label', None)
    drive_video_path = os.path.join(data_dir, video_filename)
    local_temp_path = None

    try:
        local_temp_path = copy_to_local_cache(drive_video_path, local_cache_dir)
        frames = video_processor(video_path=local_temp_path, frame_skip=frame_skip, visualize=visualize)
        if frames:
            sequence = np.stack(frames)
        else:
            sequence = np.zeros((0, feature_dim), dtype=np.float32)

        final_seq = pad_or_truncate_sequence(sequence, max_sequence_length, feature_dim)

        ensure_dir(npy_output_dir)
        base_name = os.path.basename(video_filename)
        npy_name = base_name.replace('.mp4', '.npy').replace('.avi', '.npy')
        npy_path = os.path.join(npy_output_dir, npy_name)
        np.save(npy_path, final_seq)

        rel_path = os.path.relpath(npy_path, processed_data_root)
        return {'npy_path': rel_path, 'label': label}

    except Exception as e:
        print(f"[ERROR] Processing {video_filename}: {e}")
        return None

    finally:
        if local_temp_path and os.path.exists(local_temp_path):
            os.remove(local_temp_path)


def process_dataframe_to_npy(
    df: pd.DataFrame,
    dataset_name: str,
    video_processor: Callable[..., List[np.ndarray]],
    config: ProcessingConfig = ProcessingConfig(),
    frame_skip: int = 3,
    visualize: bool = False,
    progress: bool = True
) -> pd.DataFrame:
    """
    Process rows and save .npy files. Uses config to determine paths and sizes.
    """
    data_dir = config.data_dir
    processed_data_dir = config.processed_data_dir
    local_cache_dir = config.local_cache_dir
    max_sequence_length = config.max_sequence_length
    feature_dim = config.feature_dim

    npy_output_dir = os.path.join(processed_data_dir, dataset_name)
    ensure_dir(npy_output_dir)
    ensure_dir(local_cache_dir)

    metadata: List[Dict] = []
    iterator = df.iterrows()
    if progress:
        iterator = tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}")

    for _, row in iterator:
        meta = process_single_video_row(
            row=row,
            data_dir=data_dir,
            npy_output_dir=npy_output_dir,
            processed_data_root=processed_data_dir,
            local_cache_dir=local_cache_dir,
            video_processor=video_processor,
            frame_skip=frame_skip,
            max_sequence_length=max_sequence_length,
            feature_dim=feature_dim,
            visualize=visualize
        )
        if meta is not None:
            metadata.append(meta)

    meta_df = pd.DataFrame(metadata)
    meta_csv_path = os.path.join(processed_data_dir, f"{dataset_name}_processed_metadata.csv")
    ensure_dir(processed_data_dir)
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"[INFO] Saved metadata for {dataset_name} -> {meta_csv_path}")
    return meta_df
