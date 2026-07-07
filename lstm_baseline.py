"""
LSTM Baseline - TST 성능 비교용
================================
동일한 데이터/조건에서 LSTM 성능을 측정하여 TST와 비교
(데이터 준비 코드는 cwru_tst.py의 STEP 1~2와 동일)

공정 비교를 위해 다음 조건을 TST와 동일하게 유지:
- 데이터 (윈도우 1024, stride 512)
- 분할 비율 (7:1.5:1.5)
- batch_size (64)
- optimizer (Adam, lr=0.001)
- epochs (20)
→ 오직 "모델 구조"만 다르게
"""

import torch
import torch.nn as nn


# ============================================================
# LSTM 모델 정의
# ============================================================
class LSTMClassifier(nn.Module):
    """LSTM 기반 베어링 결함 분류 (TST 비교용 baseline)"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,      # 진동값 1개
            hidden_size=hidden_size,    # 은닉 상태 64차원
            num_layers=num_layers,      # LSTM 2층
            batch_first=True,           # (batch, seq, feature)
            dropout=0.1
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1024, 1)
        out, (h_n, c_n) = self.lstm(x)   # out: (batch, 1024, 64)
        out = out[:, -1, :]              # 마지막 시점만 사용 (batch, 64)
        return self.classifier(out)      # (batch, 4)


# ============================================================
# 학습 (TST와 동일 조건)
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_lstm = LSTMClassifier().to(device)
print(model_lstm)

optimizer = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
EPOCHS = 20

# 주의: train_loader, val_loader, n_train, n_val 은
#       cwru_tst.py의 데이터 준비 코드를 먼저 실행해야 함
for epoch in range(EPOCHS):
    model_lstm.train()
    train_loss, train_correct = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model_lstm(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (pred.argmax(1) == yb).sum().item()

    model_lstm.eval()
    val_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_lstm(xb)
            val_correct += (pred.argmax(1) == yb).sum().item()

    print(f"Epoch {epoch+1} | "
          f"Train Acc: {train_correct/n_train:.4f} | "
          f"Val Acc: {val_correct/n_val:.4f}")
