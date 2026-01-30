# RevoScope AST Training Guide

This guide will help you retrain the respiratory sound classifier using the Audio Spectrogram Transformer (AST).

## Prerequisites

1. **Download ICBHI 2017 Dataset**
   - Visit: https://bhichallenge.med.auth.gr/
   - Register for access
   - Download `ICBHI_final_database.zip`
   - Extract to: `backend/data/icbhi/`

2. **Install Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

## Quick Start

### Option 1: Default Training (Recommended)
```bash
python train_ast_model.py
```

This will:
- Use ICBHI dataset from `./data/icbhi/`
- Train for 50 epochs with early stopping
- Use mixup augmentation
- Save best model to `./weights/respiratory_ast_best.pth`

### Option 2: Custom Configuration
```bash
python train_ast_model.py \
    --data_dir ./data/icbhi \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --freeze_epochs 15 \
    --patience 15
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./data/icbhi` | Path to ICBHI dataset |
| `--epochs` | `50` | Maximum training epochs |
| `--batch_size` | `16` | Batch size (reduce if OOM) |
| `--lr` | `1e-4` | Initial learning rate |
| `--freeze_epochs` | `10` | Epochs to freeze backbone |
| `--val_ratio` | `0.2` | Validation split ratio |
| `--patience` | `10` | Early stopping patience |
| `--mixup` | `True` | Use mixup augmentation |
| `--output_dir` | `./weights` | Save directory |

## Expected Training Time

- **With GPU (CUDA/MPS):** 15-30 minutes
- **With CPU:** 1-2 hours

## Output Files

After training, you'll find:

```
backend/weights/
├── respiratory_ast_best.pth       # Best model checkpoint
├── confusion_matrix.png           # Classification confusion matrix
└── training_history.png           # Loss/accuracy plots
```

## Using the Trained Model

To use the new AST model in your API, you'll need to update `api_server.py`:

1. Replace the `EfficientNetB2_LSTM` class with `ASTClassifier`
2. Update the feature extraction to use AST format
3. Point `MODEL_PATH` to the new checkpoint

(This will be handled automatically in the next step)

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python train_ast_model.py --batch_size 8
```

### Dataset Not Found
Make sure ICBHI dataset is extracted to the correct location:
```
backend/data/icbhi/
├── *.wav (audio files)
├── respiratory_cycle_annotations.txt
└── patient_diagnosis.csv
```

### Slow Training
Enable GPU acceleration:
- For NVIDIA: Install CUDA-enabled PyTorch
- For Apple Silicon: PyTorch with MPS support (already enabled)

## Next Steps

After training completes:
1. Check validation accuracy in console output
2. Review confusion matrix to see per-class performance
3. Update `api_server.py` to use the new model
4. Test inference on sample audio files

---

For questions or issues, refer to the implementation plan or contact the dev team.
