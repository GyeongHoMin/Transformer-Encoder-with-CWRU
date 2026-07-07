# Transformer Encoder-Based Bearing Fault Diagnosis

A Time Series Transformer (TST) for classifying bearing faults from raw vibration signals using the CWRU dataset.

**Author:** Gyeong Ho Min, Dept. of Mechanical Engineering, Hankyong National University

---

## Overview

This project implements a Transformer Encoder to classify 4 bearing conditions directly from raw vibration signals, without hand-crafted feature engineering. Self-attention captures long-range dependencies in the vibration sequence, outperforming a conventional LSTM baseline under identical conditions.

| Class | Description |
|-------|-------------|
| Normal (N) | Healthy bearing |
| Ball (B) | Ball defect |
| Inner Race (IR) | Inner race defect |
| Outer Race (OR) | Outer race defect |

---

## Results

| Metric | TST | LSTM |
|--------|-----|------|
| Test Accuracy | **96%** | — |
| Macro F1-score | **0.96** | — |
| Best Val Accuracy | **97.75%** (Epoch 10) | 90.64% (Epoch 20) |
| Convergence speed | ~1 epoch | ~14 epochs |

Per-class F1: Normal 1.00 / Ball 0.92 / IR 0.96 / OR 0.95

The confusion matrix shows Normal is perfectly separated (503/503), while minor confusion occurs among the three fault types (Ball/IR/OR), whose vibration signatures are relatively similar.

---

## Pipeline

```
Raw vibration signal (.mat, 40 files)
   ↓  Sliding window (size=1024, stride=512, 50% overlap) → 11,832 windows
   ↓  Z-score normalization (StandardScaler)
   ↓  Tensor conversion + Train/Val/Test split (70/15/15)
   ↓  DataLoader (batch_size=64, shuffle)
   ↓  input_proj: Linear(1 → 64)
   ↓  + positional embedding
   ↓  Transformer Encoder × 3 (Multi-Head Attention + FFN)
   ↓  Global Average Pooling
   ↓  Classifier: Linear(64 → 4)
Fault diagnosis result
```

---

## Model Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Window size | 1024 | ~2.5 bearing revolutions (physical basis) |
| Stride | 512 | 50% overlap |
| d_model | 64 | embedding dimension |
| nhead | 8 | attention heads (64 ÷ 8 = 8) |
| dim_feedforward | 256 | d_model × 4 (Transformer convention) |
| num_layers | 3 | encoder depth |
| dropout | 0.1 | regularization |
| optimizer | Adam (lr=1e-3) | with StepLR (step=5, gamma=0.5) |
| epochs | 20 | batch size 64 |

---

## Files

| File | Description |
|------|-------------|
| `cwru_tst.py` | Full pipeline: data loading, preprocessing, TST model, training, evaluation |
| `lstm_baseline.py` | LSTM model for performance comparison |

---

## Usage

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn

# Set your dataset path in cwru_tst.py, then:
python cwru_tst.py
```

**Dataset:** [CWRU Bearing Dataset (Kaggle)](https://www.kaggle.com/datasets/astrollama/cwru-case-western-reserve-university-dataset)

Only the Drive End (DE) channel is used, as the fault signal is most pronounced there.

---

## Key Design Choices

- **Encoder-only architecture** — classification requires no sequence generation, so no decoder is needed (like BERT).
- **Global Average Pooling instead of CLS token** — simpler to implement while achieving comparable accuracy.
- **No outlier removal** — in vibration signals, large-amplitude values represent fault information, not noise.
- **Random split** — valid here because the task is classification (not time-series forecasting) and classes are balanced (~24–28% each).

---

## Future Work

- Early Stopping to automatically select the optimal epoch (Val Acc peaks at Epoch 10, mild overfitting after).
- Cross-load generalization: train on some load conditions (0/1/2 hp) and test on an unseen load (3 hp).
- File-level split to eliminate weak information sharing from overlapping windows.
- Hyperparameter tuning (grid search over d_model, num_layers).
