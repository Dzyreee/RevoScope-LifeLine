"""
Dataset utilities for ICBHI 2017 respiratory sound classification
Includes downloading, preprocessing, and augmentation functions
"""

import os
import requests
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from scipy.ndimage import zoom
import warnings

warnings.filterwarnings('ignore')


# Dataset configuration
ICBHI_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/"
DATASET_FILES = [
    "ICBHI_final_database.zip",
    "patient_diagnosis.csv"
]

CLASS_MAPPING = {
    'normal': 0,
    'crackle': 1, 
    'wheeze': 2,
    'both': 3
}


def download_icbhi_dataset(data_dir='./data/icbhi'):
    """
    Download ICBHI 2017 dataset
    Note: The actual ICBHI dataset requires registration. This is a placeholder.
    You'll need to manually download from: https://bhichallenge.med.auth.gr/
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ICBHI 2017 Dataset Download")
    print("=" * 60)
    print("\nIMPORTANT: The ICBHI dataset requires manual registration.")
    print("\nPlease follow these steps:")
    print("1. Visit: https://bhichallenge.med.auth.gr/")
    print("2. Register and download 'ICBHI_final_database.zip'")
    print(f"3. Extract the contents to: {data_path.absolute()}")
    print("\nExpected structure:")
    print(f"  {data_path}/")
    print("    ├── audio/")
    print("    ├── patient_diagnosis.csv")
    print("    └── respiratory_cycle_annotations.txt")
    print("\n" + "=" * 60)
    
    return data_path


def load_icbhi_metadata(data_dir='./data/icbhi'):
    """Load ICBHI metadata from individual annotation files"""
    data_path = Path(data_dir)
    
    # Check if dataset exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please run download_icbhi_dataset() first."
        )
    
    # Get all .wav files
    wav_files = list(data_path.glob('*.wav'))
    
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in {data_path}. "
            "Please download the ICBHI dataset."
        )
    
    print(f"Found {len(wav_files)} audio files")
    
    # Parse individual annotation files
    annotations = []
    
    for wav_file in wav_files:
        # Get corresponding .txt file
        txt_file = wav_file.with_suffix('.txt')
        
        if not txt_file.exists():
            # Skip if no annotation file
            continue
        
        # Parse annotation file
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 4:
                    try:
                        annotations.append({
                            'filename': wav_file.stem,  # filename without extension
                            'start': float(parts[0]),
                            'end': float(parts[1]),
                            'crackles': int(parts[2]),
                            'wheezes': int(parts[3])
                        })
                    except ValueError:
                        # Skip malformed lines
                        continue
    
    if not annotations:
        raise ValueError("No valid annotations found in .txt files")
    
    df = pd.DataFrame(annotations)
    
    # Create class labels
    df['class'] = df.apply(lambda x: classify_sound(x['crackles'], x['wheezes']), axis=1)
    
    print(f"Loaded {len(df)} respiratory cycles from {len(wav_files)} patients")
    
    return df


def classify_sound(crackles, wheezes):
    """Classify sound based on presence of crackles and wheezes"""
    if crackles > 0 and wheezes > 0:
        return 'both'
    elif crackles > 0:
        return 'crackle'
    elif wheezes > 0:
        return 'wheeze'
    else:
        return 'normal'


def create_sample_metadata(data_path):
    """Create sample metadata for testing if real dataset not available"""
    print("Creating sample metadata structure...")
    
    # Look for any audio files
    audio_files = list(data_path.glob('*.wav'))
    
    if not audio_files:
        raise FileNotFoundError(
            f"No .wav files found in {data_path}. "
            "Please download the ICBHI dataset."
        )
    
    # Create basic metadata
    metadata = []
    for audio_file in audio_files:
        metadata.append({
            'filename': audio_file.stem,
            'start': 0.0,
            'end': 5.0,
            'crackles': 0,
            'wheezes': 0,
            'class': 'normal'
        })
    
    return pd.DataFrame(metadata)


def extract_features_ast(audio_path, target_length=16000*5, sample_rate=16000):
    """
    Extract log-mel spectrogram for AST model
    Optimized: 10ms hop length, 512 frames (5.12s)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Pad or trim to target length in samples
        # 512 frames * 160 hop = 81920 samples ~ 5.12s
        target_len_samples = 512 * 160
        if len(y) < target_len_samples:
            y = np.pad(y, (0, target_len_samples - len(y)))
        else:
            y = y[:target_len_samples]
        
        # Compute mel spectrogram
        # Standard AST: 25ms window, 10ms shift
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=400,          # 25ms
            hop_length=160,     # 10ms
            n_mels=128,
            fmin=50,
            fmax=4000
        )
        
        # Pad/crop to exactly 512 frames
        target_frames = 512
        n_mels, n_frames = mel_spec.shape
        if n_frames < target_frames:
            padding = target_frames - n_frames
            mel_spec = np.pad(mel_spec, ((0, 0), (0, padding)), mode='constant')
        else:
            mel_spec = mel_spec[:, :target_frames]
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        return log_mel.T.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return zeros as fallback
        return np.zeros((512, 128), dtype=np.float32)


def augment_audio(y, sr=16000):
    """
    Apply audio augmentation techniques
    
    Args:
        y: Audio signal
        sr: Sample rate
        
    Returns:
        Augmented audio signal
    """
    augmentations = []
    
    # Original
    augmentations.append(y)
    
    # Time stretch
    if np.random.rand() > 0.5:
        rate = np.random.uniform(0.9, 1.1)
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        augmentations.append(y_stretched)
    
    # Pitch shift
    if np.random.rand() > 0.5:
        n_steps = np.random.randint(-2, 3)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        augmentations.append(y_shifted)
    
    # Add noise
    if np.random.rand() > 0.5:
        noise = np.random.randn(len(y)) * 0.005
        y_noisy = y + noise
        augmentations.append(y_noisy)
    
    # Return random augmentation
    return augmentations[np.random.randint(0, len(augmentations))]


def spec_augment(mel_spec, time_mask_param=20, freq_mask_param=10):
    """
    Apply SpecAugment to mel spectrogram (Time, Freq)
    
    Args:
        mel_spec: Mel spectrogram (time, freq)
        time_mask_param: Max time mask length
        freq_mask_param: Max frequency mask length
        
    Returns:
        Augmented mel spectrogram
    """
    mel_spec = mel_spec.copy()
    n_time, n_freq = mel_spec.shape
    
    # Time masking (masking rows)
    if np.random.rand() > 0.5:
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, n_time - t)
        mel_spec[t0:t0+t, :] = 0
    
    # Frequency masking (masking cols)
    if np.random.rand() > 0.5:
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, n_freq - f)
        mel_spec[:, f0:f0+f] = 0
    
    return mel_spec


def mixup_data(x1, y1, x2, y2, alpha=0.4):
    """
    Apply mixup augmentation
    
    Args:
        x1, x2: Input spectrograms
        y1, y2: Labels
        alpha: Mixup strength
        
    Returns:
        Mixed spectrogram and label
    """
    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y1 + (1 - lam) * y2
    
    return x_mixed, y_mixed


def get_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        labels: Array of class labels
        
    Returns:
        Dictionary of class weights
    """
    from collections import Counter
    
    counts = Counter(labels)
    total = len(labels)
    
    weights = {}
    for cls, count in counts.items():
        weights[cls] = total / (len(counts) * count)
    
    return weights


def train_val_split(metadata, val_ratio=0.2, random_state=42):
    """
    Split dataset into train and validation sets
    Stratified by class to maintain distribution
    
    Args:
        metadata: DataFrame with metadata
        val_ratio: Validation set ratio
        random_state: Random seed
        
    Returns:
        Train and validation DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        metadata,
        test_size=val_ratio,
        stratify=metadata['class'],
        random_state=random_state
    )
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


if __name__ == "__main__":
    # Test the utilities
    print("Testing dataset utilities...")
    
    # Download dataset (will show instructions)
    data_path = download_icbhi_dataset()
    
    print("\nDataset utilities ready!")
    print("Remember to download the ICBHI dataset manually.")
