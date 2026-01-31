"""
RevoScope - Optimized Production API Server
AST (Audio Spectrogram Transformer) Inference
"""

import os
import tempfile
import numpy as np
from typing import Optional, Dict
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import warnings
from transformers import ASTForAudioClassification, AutoFeatureExtractor

# Suppress warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_PATH = WEIGHTS_DIR / "respiratory_ast_best.pth"

# Audio parameters
SAMPLE_RATE = 16000
DURATION = 5  # AST expects fixed input length usually around 10s, but we'll pad/truncate
TARGET_SAMPLE_RATE = 16000

CLASS_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']

# Global model and processor
model = None
feature_extractor = None
device = None

def load_model():
    """Load AST model with custom classifier matching saved weights."""
    global model, feature_extractor, device
    
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Loading AST model on {device}...")

    # Initialize Feature Extractor (standard AST config but forced max_length)
    try:
        # Force max_length=510 during init to match model
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            max_length=510
        )
    except Exception as e:
        print(f"Warning: Could not load feature extractor from Hub: {e}")
        feature_extractor = None 

    # Initialize Model Architecture
    try:
        # Use custom config matching saved weights (max_length=500 -> 602 position embeddings)
        from transformers import ASTConfig
        config = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        config.num_labels = 4
        config.max_length = 510  # Matches saved weights position embeddings (602 patches)
        
        model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Replace classifier with custom Sequential matching saved weights
        # Saved weights structure: classifier.0 (LayerNorm), 2 (Linear 768->256), 5 (Linear 256->4)
        import torch.nn as nn
        model.classifier = nn.Sequential(
            nn.LayerNorm(768),      # classifier.0
            nn.GELU(),              # classifier.1 (activation, no weights)
            nn.Linear(768, 256),    # classifier.2
            nn.Dropout(0.1),        # classifier.3 (no weights)
            nn.GELU(),              # classifier.4 (activation, no weights)
            nn.Linear(256, 4)       # classifier.5
        )
        print("Custom classifier head installed (768→256→4)")
        
        # Load weights
        if MODEL_PATH.exists():
            if MODEL_PATH.stat().st_size < 10000:
                print(f"WARNING: Model file is too small ({MODEL_PATH.stat().st_size} bytes). Git LFS pointer?")
                print("Please run 'git lfs pull' to download the real model.")
            else:
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, val_acc: {checkpoint.get('val_acc', 0):.2%})")
                else:
                    state_dict = checkpoint
                
                # Remap keys: 'ast.' -> 'audio_spectrogram_transformer.'
                remapped_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('ast.', 'audio_spectrogram_transformer.')
                    remapped_state_dict[new_key] = value
                print(f"Remapped {len(remapped_state_dict)} keys (ast. → audio_spectrogram_transformer.)")
                    
                msg = model.load_state_dict(remapped_state_dict, strict=False)
                print(f"Weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        else:
            print(f"WARNING: Model file not found at {MODEL_PATH}")

        model.to(device)
        model.eval()
        print("✓ AST Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading AST model: {e}")
        import traceback
        traceback.print_exc()
        model = None


# FastAPI Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="RevoScope API (AST)", version="3.0", lifespan=lifespan)

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
    return {"status": "healthy", "model_loaded": model is not None, "device": str(device)}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(500, "AST Model not loaded. check server logs.")
    
    suffix = Path(audio.filename).suffix or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    
    try:
        # Load and Preprocess Audio
        y, sr = librosa.load(tmp_path, sr=TARGET_SAMPLE_RATE, mono=True)
        
        # SLIDING WINDOW ANALYSIS
        # AST needs ~5.12s inputs. We'll use 5s overlapping windows.
        WINDOW_SIZE = TARGET_SAMPLE_RATE * 5  # 5 seconds
        STRIDE = int(WINDOW_SIZE * 0.5)       # 50% overlap (2.5s)
        
        # If audio is shorter than window, pad it
        if len(y) < WINDOW_SIZE:
            y = np.pad(y, (0, WINDOW_SIZE - len(y)))
            
        # Create windows
        windows = []
        for i in range(0, len(y) - WINDOW_SIZE + 1, STRIDE):
            window = y[i:i + WINDOW_SIZE]
            windows.append(window)
            
        # If no windows (shouldn't happen due to padding), use whole clip
        if not windows:
            windows = [y]
            
        print(f"Processing {len(windows)} sliding windows...")
        
        # Batch Processing
        batch_probs = []
        
        for window in windows:
            # Normalize window (Zero Mean, Unit Variance)
            if len(window) > 0:
                # Add small epsilon to avoid divide by zero for silent clips
                window = (window - np.mean(window)) / (np.std(window) + 1e-6)
            
            if feature_extractor:
                inputs = feature_extractor(
                    window, 
                    sampling_rate=TARGET_SAMPLE_RATE, 
                    return_tensors="pt",
                    max_length=510,
                    truncation=True,
                    padding="max_length"
                )
                input_values = inputs.input_values.to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(input_values)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    batch_probs.append(probs)
            else:
                raise HTTPException(500, "Feature extractor not initialized")

        # AGGREGATION (Max Pooling strategy)
        # We take the maximum probability observed for each class across all windows.
        # This ensures that if a Wheeze is heard clearly in *one* window, it's not washed out by 10 windows of silence.
        batch_probs = np.array(batch_probs)
        final_probs = np.max(batch_probs, axis=0)
        
        # Re-normalize to sum to 1
        final_probs = final_probs / (np.sum(final_probs) + 1e-6)
        
        class_id = int(np.argmax(final_probs))
        confidence = float(final_probs[class_id] * 100)
        class_name = CLASS_NAMES[class_id]
        
        # Severity Logic
        severity_map = {'Normal': 15, 'Crackle': 55, 'Wheeze': 50, 'Both': 75}
        severity = int(severity_map.get(class_name, 50) + (1 - confidence/100) * 25)
        severity = min(100, max(0, severity))
        
        # ESI Logic
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
            2: f"Significant {class_name.lower()}. Urgent evaluation.",
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
            heart_rate=None # Removed HR detection for speed/AST focus for now
        )
            
    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 40)
    print("RevoScope AST API Server v3.0")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000)
