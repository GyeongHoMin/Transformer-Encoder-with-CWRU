"""
CWRU 베어링 결함 분류 - Time Series Transformer (TST)
========================================================
원시 진동 신호로부터 4종 베어링 결함(Normal/Ball/IR/OR)을 분류하는
Transformer Encoder 기반 딥러닝 모델

Author: Gyeong Ho Min (한경국립대학교 기계공학과)
Dataset: CWRU Bearing Dataset
"""

import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# STEP 1. 데이터 로딩 + 슬라이딩 윈도우
# ============================================================
# 데이터 경로: Kaggle에서 CWRU 데이터셋을 받아 이 폴더에 .mat 파일들을 넣으세요
# https://www.kaggle.com/datasets/astrollama/cwru-case-western-reserve-university-dataset
path = "./data/cwru"   # 예: 프로젝트 폴더 아래 data/cwru/ 에 .mat 파일 배치

# 파일 이름 → 클래스 라벨 매핑
label_map = {'Normal': 0, 'B': 1, 'IR': 2, 'OR': 3}


def get_label(filename):
    """파일 이름 앞부분으로 클래스 라벨(0~3) 반환"""
    for key in label_map:
        if filename.startswith(key):
            return label_map[key]
    return None


def get_de_key(mat):
    """.mat 파일에서 Drive End 진동 신호 키 탐색"""
    for k in mat.keys():
        if 'DE_time' in k:
            return k
    return None


# 슬라이딩 윈도우 설정
WINDOW_SIZE = 1024   # 윈도우 크기 (베어링 약 2.5회전 분량)
STRIDE = 512         # 이동 간격 (50% 겹침)

X, y = [], []

for fname in sorted(os.listdir(path)):
    if not fname.endswith('.mat'):
        continue
    label = get_label(fname)
    if label is None:
        continue

    mat = scipy.io.loadmat(os.path.join(path, fname))
    de_key = get_de_key(mat)
    if de_key is None:
        continue

    signal = mat[de_key].flatten()   # (N, 1) → (N,) 1차원으로

    # 1024개씩 자르되 512씩 이동 (50% 겹침)
    for start in range(0, len(signal) - WINDOW_SIZE, STRIDE):
        window = signal[start:start + WINDOW_SIZE]
        X.append(window)
        y.append(label)

X = np.array(X)   # (11832, 1024)
y = np.array(y)   # (11832,)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"클래스 분포: {np.bincount(y)}")   # [3310, 2840, 2838, 2844]


# ============================================================
# STEP 2. 정규화 + 텐서 변환 + 데이터 분할
# ============================================================

# Z-score 정규화 (평균 0, 표준편차 1)
# 클래스마다 진동 크기가 달라 값의 크기가 아닌 패턴으로 학습하게 함
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # (11832, 1024)

# numpy → PyTorch 텐서 변환
# unsqueeze(-1): (11832, 1024) → (11832, 1024, 1)  [TST 입력 형식]
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1)
y_tensor = torch.tensor(y, dtype=torch.long)   # 라벨은 정수형

# Train/Val/Test 분할 (7 : 1.5 : 1.5)
dataset = TensorDataset(X_tensor, y_tensor)
n = len(dataset)
n_train = int(n * 0.7)
n_val = int(n * 0.15)
n_test = n - n_train - n_val

train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

# DataLoader: 64개씩 배치로 묶기 (train만 매 epoch 섞기)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")


# ============================================================
# STEP 3. TST 모델 정의
# ============================================================
class TST(nn.Module):
    """Time Series Transformer for Bearing Fault Classification"""

    def __init__(self, seq_len=1024, d_model=64, nhead=8,
                 num_layers=3, num_classes=4, dropout=0.1):
        super().__init__()

        # 입력 임베딩: 진동값 1개 → 64차원 벡터로 확장
        self.input_proj = nn.Linear(1, d_model)

        # 위치 인코딩 (학습 가능한 파라미터)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer Encoder (3층)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,          # 임베딩 차원 = 64
            nhead=nhead,              # Attention 헤드 수 = 8
            dim_feedforward=256,      # FFN 중간 차원 (d_model × 4)
            dropout=dropout,          # 과적합 방지 0.1
            batch_first=True          # (batch, seq, feature) 형식
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)   # 64 → 4 클래스
        )

    def forward(self, x):
        # x: (batch, 1024, 1)
        x = self.input_proj(x)          # (batch, 1024, 64)
        x = x + self.pos_embedding      # 위치 정보 추가
        x = self.transformer(x)         # Self-Attention × 3층
        x = x.mean(dim=1)               # GAP: (batch, 1024, 64) → (batch, 64)
        return self.classifier(x)       # (batch, 4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TST().to(device)
print(model)


# ============================================================
# STEP 4. 학습
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

EPOCHS = 20

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss, train_correct = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()           # gradient 초기화
        pred = model(xb)                # 순전파
        loss = criterion(pred, yb)      # 손실 계산
        loss.backward()                 # 역전파
        optimizer.step()                # 파라미터 업데이트
        train_loss += loss.item()
        train_correct += (pred.argmax(1) == yb).sum().item()

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
            val_correct += (pred.argmax(1) == yb).sum().item()

    scheduler.step()

    print(f"Epoch {epoch+1:02d} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_correct/n_train:.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_correct/n_val:.4f}")

# 모델 저장
torch.save(model.state_dict(), 'tst_model.pth')
print("모델 저장 완료: tst_model.pth")


# ============================================================
# STEP 5. 테스트 평가 + 혼동행렬
# ============================================================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(yb.numpy())

# 성능 리포트 (Precision / Recall / F1)
print(classification_report(all_labels, all_preds,
      target_names=['Normal', 'Ball', 'IR', 'OR']))

# 혼동행렬 시각화
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Normal', 'Ball', 'IR', 'OR'],
            yticklabels=['Normal', 'Ball', 'IR', 'OR'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=180)
plt.show()
