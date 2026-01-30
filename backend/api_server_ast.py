"""
RevoScope - AST-Powered API Server
Audio Spectrogram Transformer inference with optimized preprocessing

Run: python api_server_ast.py
"""

import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import ASTModel
from contextlib import asynccontextmanager
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Paths
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_PATH_AST = WEIGHTS_DIR / "respiratory_ast_best.pth"
MODEL_PATH_OLD = WEIGHTS_DIR / "respiratory_final.pth"

# Audio parameters
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
TARGET_LENGTH = SAMPLE_RATE * DURATION

CLASS_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']

# Global model and device
model = None
device = None
model_type = None  # 'ast' or 'efficientnet'


class ASTClassifier(nn.Module):
    """Audio Spectrogram Transformer for respiratory classification"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Load AST backbone (will fail if transformers not installed)
        # Initialize AST with correct config (512 frames)
        try:
            from transformers import ASTConfig, ASTModel
            # Load config from AudioSet but override length
            config = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            config.max_length = 512
            self.ast = ASTModel(config)
        except:
            # Fallback
            from transformers import ASTConfig, ASTModel
            config = ASTConfig()
            self.ast = ASTModel(config)
        
        hidden_size = self.ast.config.hidden_size  # 768
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """Forward pass expecting (batch, freq, time)"""
        outputs = self.ast(x)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return logits


def extract_features_ast(audio_path):
    """
    Extract log-mel spectrogram for AST model
    Optimized: 10ms hop length, 512 frames (5.12s)
    """
    try:
        # Load audio
        y, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
    except:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # Pad or trim to target length (512 frames * 160 hop)
    target_len_samples = 512 * 160
    if len(y) < target_len_samples:
        y = np.pad(y, (0, target_len_samples - len(y)))
    else:
        y = y[:target_len_samples]
    
    # Compute mel spectrogram (standard AudioSet params)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=400,          # 25ms
        hop_length=160,     # 10ms
        n_mels=N_MELS,
        fmin=50,
        fmax=4000
    )
    
    # Pad/Crop to exactly 512 frames
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


def detect_heart_rate(audio_path, sr=SAMPLE_RATE):
    """Detect heart rate from audio using FFT and bandpass filtering"""
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, duration=DURATION)
        
        # Bandpass filter for heartbeat frequencies (60-200 BPM = 1-3.33 Hz)
        nyquist = sr / 2
        low_freq = 0.5  # 30 BPM minimum
        high_freq = 3.5  # 210 BPM maximum
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = butter(4, [low, high], btype='band')
        y_filtered = filtfilt(b, a, y)
        
        # Compute power spectral density
        freqs = np.fft.rfftfreq(len(y_filtered), d=1/sr)
        spectrum = np.abs(np.fft.rfft(y_filtered)) ** 2
        
        # Focus on heartbeat frequency range
        freq_mask = (freqs >= 0.5) & (freqs <= 3.5)
        freqs_masked = freqs[freq_mask]
        spectrum_masked = spectrum[freq_mask]
        
        if len(spectrum_masked) == 0:
            return None
        
        # Find dominant frequency
        peaks, properties = find_peaks(spectrum_masked, height=0)
        
        if len(peaks) == 0:
            dominant_idx = np.argmax(spectrum_masked)
        else:
            peak_heights = properties['peak_heights']
            dominant_idx = peaks[np.argmax(peak_heights)]
        
        dominant_freq = freqs_masked[dominant_idx]
        heart_rate = int(round(dominant_freq * 60))
        
        # Validate heart rate
        if 40 <= heart_rate <= 200:
            return heart_rate
        else:
            return None
            
    except Exception as e:
        print(f"Heart rate detection error: {e}")
        return None


def load_model():
    """Load model (AST if available, otherwise fallback to old model)"""
    global model, device, model_type
    
    # Setup device
    try:
        # DirectML disabled due to stability issues
        # import torch_directml
        # device = torch_directml.device()
        # print(f"✓ Using DirectML (AMD GPU acceleration)")
        raise ImportError("DirectML disabled")
    except ImportError as e:
        # print(f"⚠ DirectML not found: {e}")
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"✓ Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ Using CUDA (NVIDIA)")
        else:
            device = torch.device('cpu')
            print(f"✓ Using CPU (no GPU acceleration)")
    
    # Try to load AST model first
    if MODEL_PATH_AST.exists():
        print(f"Loading AST model from {MODEL_PATH_AST}...")
        try:
            # ASTClassifier already initializes with the correct 512-frame config
            model = ASTClassifier(num_classes=4)
            checkpoint = torch.load(MODEL_PATH_AST, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_type = 'ast'
            val_acc = checkpoint.get('val_acc', 0)
            print(f"✓ AST model loaded ({val_acc*100:.1f}% accuracy) on {device}")
        except Exception as e:
            print(f"Failed to load AST model: {e}")
            print("Falling back to EfficientNet model...")
            model = None
    
    # Fallback to old model if AST not available
    if model is None and MODEL_PATH_OLD.exists():
        print(f"Loading EfficientNet model from {MODEL_PATH_OLD}...")
        from api_server import EfficientNetB2_LSTM
        model = EfficientNetB2_LSTM(num_classes=4)
        checkpoint = torch.load(MODEL_PATH_OLD, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_type = 'efficientnet'
        val_acc = checkpoint.get('val_acc', 0)
        print(f"✓ EfficientNet model loaded ({val_acc*100:.1f}% accuracy) on {device}")
    
    if model is None:
        raise FileNotFoundError(
            f"No model found. Please train a model first or ensure model files exist:\n"
            f"  AST: {MODEL_PATH_AST}\n"
            f"  EfficientNet: {MODEL_PATH_OLD}"
        )
    
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)


# FastAPI with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="RevoScope API (AST)",
    description="AI respiratory sound classification with AST",
    version="3.0.0",
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
    heart_rate: int | None = None
    model_type: str | None = None


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "device": str(device)
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(audio: UploadFile = File(...)):
    """Audio analysis endpoint with AST or EfficientNet"""
    if model is None:
        raise HTTPException(500, "Model not loaded")
    
    # Save temp file
    suffix = Path(audio.filename).suffix or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    
    try:
        # Extract features based on model type
        if model_type == 'ast':
            features = extract_features_ast(tmp_path)
            x = torch.from_numpy(features).unsqueeze(0).to(device)
        else:
            # Use old feature extraction
            from api_server import extract_features_fast
            features = extract_features_fast(tmp_path)
            x = torch.from_numpy(features).unsqueeze(0).to(device)
        
        # Detect heart rate
        heart_rate = detect_heart_rate(tmp_path)
        
        # Inference
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
            heart_rate=heart_rate,
            model_type=model_type
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 40)
    print("RevoScope AST API Server v3.0")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000)
