# RevoScope AI Backend

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model (One-time setup)
```bash
python train_model.py
```

This will automatically:
- Download ICBHI 2017 dataset (~1.5GB)
- Download pre-trained PANNs weights (~300MB)
- Train the respiratory classifier (~30-60 min)
- Save model to `weights/respiratory_classifier.pth`

### 3. Start the API Server
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
Health check.

## Model Architecture

- **Input**: 128-mel spectrogram (5-second segments)
- **Model**: 4-layer CNN with BatchNorm
- **Output**: 4 classes (Normal, Crackle, Wheeze, Both)
- **Training Data**: ICBHI 2017 (~6900 respiratory cycles)

## Expected Accuracy

Based on ICBHI 2017 benchmark:
- 4-class accuracy: ~80-85%
- Normal vs Abnormal: ~90%+
