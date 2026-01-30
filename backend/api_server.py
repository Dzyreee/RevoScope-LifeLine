"""
RevoScope - Optimized Production API Server
Fast EfficientNet-B2 + LSTM inference with caching

Run: python api_server.py
"""

import os
import tempfile
import numpy as np
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.ndimage import zoom
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torchvision.models as models
from contextlib import asynccontextmanager
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Paths
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_PATH = WEIGHTS_DIR / "respiratory_final.pth"

# Audio parameters - optimized for speed
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128  # Reduced from 224 for speed
N_MFCC = 40
N_FFT = 1024  # Reduced from 2048
TARGET_SIZE = 224

CLASS_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']

# Global model and device
model = None
device = None


class EfficientNetB2_LSTM(nn.Module):
    """EfficientNet-B2 + LSTM classifier with attention."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights=None)
        self.backbone.classifier = nn.Identity()
        
        self.lstm = nn.LSTM(
            input_size=1408, hidden_size=256,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone.features(x)
        features = features.permute(0, 2, 3, 1).reshape(x.size(0), -1, 1408)
        lstm_out, _ = self.lstm(features)
        attn = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn * lstm_out, dim=1)
        return self.classifier(context)


def extract_features_fast(audio_path):
    """Optimized feature extraction - single pass."""
    try:
        # Try wav first (faster)
        import soundfile as sf
        y, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        if len(y.shape) > 1:
            y = y.mean(axis=1)
    except:
        # Fallback
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # Ensure correct length
    target_len = sr * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    
    hop = int(len(y) / TARGET_SIZE)
    
    # Compute features in single batch
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, 
                                          hop_length=hop, n_mels=N_MELS,
                                          fmin=50, fmax=4000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, 
                                 n_fft=N_FFT, hop_length=hop)
    delta = librosa.feature.delta(mfcc)
    
    # Resize all to target
    def resize(x):
        return zoom(x, (TARGET_SIZE/x.shape[0], TARGET_SIZE/x.shape[1]), order=1)
    
    # Normalize and stack
    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    
    features = np.stack([
        norm(resize(log_mel)),
        norm(resize(mfcc)),
        norm(resize(delta))
    ], axis=0).astype(np.float32)
    
    return features[:, :TARGET_SIZE, :TARGET_SIZE]


def detect_heart_rate(audio_path, sr=SAMPLE_RATE):
    """
    Advanced heart rate detection using envelope extraction and autocorrelation.
    More robust than simple FFT for short, impulsive heart sounds.
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, duration=DURATION)
        if len(y) == 0:
            return None

        # 1. Bandpass filter for S1/S2 heart sound frequencies (20-120 Hz)
        # This removes most lung sounds (whooshing) and high-frequency noise
        nyquist = sr / 2
        f_low, f_high = 20, 120
        b, a = butter(3, [f_low/nyquist, f_high/nyquist], btype='band')
        y_filt = filtfilt(b, a, y)

        # 2. Envelope Extraction
        # Rectification (absolute value) + Smoothing (Low-pass filter at 10Hz)
        y_abs = np.abs(y_filt)
        b_env, a_env = butter(2, 10/nyquist, btype='low')
        envelope = filtfilt(b_env, a_env, y_abs)
        
        # Normalize envelope
        envelope = (envelope - np.mean(envelope)) / (np.std(envelope) + 1e-8)

        # 3. Autocorrelation
        # Find repeating patterns in the amplitude envelope
        n = len(envelope)
        # We only need autocorrelation for lags corresponding to 40-200 BPM
        # 40 BPM = 1.5s lag, 200 BPM = 0.3s lag
        min_lag = int(0.3 * sr)
        max_lag = int(1.5 * sr)
        
        # Optimized autocorrelation using FFT
        f_env = np.fft.rfft(envelope, n=2*n)
        acf = np.fft.irfft(f_env * np.conj(f_env))[:n]
        acf /= np.arange(n, 0, -1) # Unbiased
        acf /= acf[0] # Normalize
        
        # 4. Peak Detection in ACF
        # Search for the highest peak in the target lag range
        acf_segment = acf[min_lag:max_lag]
        if len(acf_segment) == 0:
            return None
            
        peaks, props = find_peaks(acf_segment, height=0.1, distance=int(0.2*sr))
        
        if len(peaks) > 0:
            # Pick the highest peak in the autocorrelation
            best_peak_idx = peaks[np.argmax(props['peak_heights'])]
            lag = best_peak_idx + min_lag
            
            heart_rate = int(round(60 * sr / lag))
            
            # Final sanity check
            if 40 <= heart_rate <= 200:
                print(f"✓ DSP HR Detected: {heart_rate} BPM (lag={lag})")
                return heart_rate
        
        # Fallback to simple peak counting in time domain if ACF fails
        peaks_time, _ = find_peaks(envelope, height=np.percentile(envelope, 85), distance=int(0.4*sr))
        if len(peaks_time) > 1:
            intervals = np.diff(peaks_time) / sr
            avg_interval = np.median(intervals)
            heart_rate = int(round(60 / avg_interval))
            if 40 <= heart_rate <= 200:
                print(f"✓ Fallback HR Detected: {heart_rate} BPM")
                return heart_rate

        return None
            
    except Exception as e:
        print(f"Heart rate detection error: {e}")
        return None


def load_model():
    """Load model with optimizations."""
    global model, device
    
    # Use MPS on Mac M-series, CUDA otherwise, fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = EfficientNetB2_LSTM(num_classes=4)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # Optimize for inference
    torch.set_grad_enabled(False)
    
    val_acc = checkpoint.get('val_acc', 0)
    print(f"✓ Model loaded ({val_acc*100:.1f}% accuracy) on {device}")


# FastAPI with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="RevoScope API",
    description="AI respiratory sound classification",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResponse(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    all_probabilities: dict
    severity_score: int
    esi_level: int
    esi_name: str
    recommendation: str
    heart_rate: Optional[int] = None


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(audio: UploadFile = File(...)):
    """Fast audio analysis endpoint."""
    if model is None:
        raise HTTPException(500, "Model not loaded")
    
    # Save temp file
    suffix = Path(audio.filename).suffix or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    
    try:
        # Extract features
        features = extract_features_fast(tmp_path)
        
        # Detect heart rate
        heart_rate = detect_heart_rate(tmp_path)
        print(f"DEBUG: Detected heart_rate = {heart_rate}")
        
        # Inference
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        outputs = model(x)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id] * 100)
        class_name = CLASS_NAMES[class_id]
        
        # Calculate severity and ESI
        severity_map = {'Normal': 15, 'Crackle': 55, 'Wheeze': 50, 'Both': 75}
        severity = int(severity_map.get(class_name, 50) + (1 - confidence/100) * 25)
        severity = min(100, max(0, severity))
        
        # ESI level
        if class_name == 'Normal' and confidence > 80:
            esi_level = 5
        elif class_name == 'Both' or severity >= 70:
            esi_level = 2
        elif severity >= 50:
            esi_level = 3
        elif severity >= 30:
            esi_level = 4
        else:
            esi_level = 5
        
        esi_names = {1: 'CRITICAL', 2: 'URGENT', 3: 'MODERATE', 4: 'LOW', 5: 'STABLE'}
        
        recommendations = {
            5: "Normal breath sounds. No intervention required.",
            4: f"Mild {class_name.lower()}. Follow-up in 1-2 weeks.",
            3: f"{class_name} detected. Consider bronchodilator therapy.",
            2: f"Significant {class_name.lower()}. Urgent evaluation. O2 if SpO2 < 94%.",
            1: "Critical respiratory distress. Immediate intervention."
        }
        
        return AnalysisResponse(
            class_name=class_name,
            class_id=class_id,
            confidence=round(confidence, 1),
            all_probabilities={n: round(float(p * 100), 1) for n, p in zip(CLASS_NAMES, probs)},
            severity_score=severity,
            esi_level=esi_level,
            esi_name=esi_names[esi_level],
            recommendation=recommendations[esi_level],
            heart_rate=heart_rate
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 40)
    print("RevoScope API Server v2.1 (Optimized)")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000)
