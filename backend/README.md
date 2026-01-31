# RevoScope AI Backend

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

# RevoScope AI Backend

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python api_server.py
```

Server runs at http://localhost:8000

## API Endpoints

### POST /analyze
Upload audio file for classification.

**Response:**
```json
{
  "class_name": "Wheeze",
  "class_id": 2,
  "confidence": 87.5,
  "all_probabilities": {
    "Normal": 5.2,
    "Crackle": 4.1,
    "Wheeze": 87.5,
    "Both": 3.2
  },
  "severity_score": 55,
  "esi_level": 3,
  "esi_name": "MODERATE",
  "recommendation": "Wheeze detected. Expedited assessment recommended."
}
```

### GET /health
Health check. Returns model loading status and device (MPS/CUDA/CPU).

## Model Architecture

- **Model**: Audio Spectrogram Transformer (AST)
- **Base**: `MIT/ast-finetuned-audioset-10-10-0.4593`
- **Fine-tuning**: Custom classification head (Sequential)
- **Input**: 510-length spectrogram sequences (normalized)
- **Output**: 4 classes (Normal, Crackle, Wheeze, Both)

## Performance
- Optimized for Apple Silicon (MPS)
- Inference speed: ~0.1-0.3s per sample
