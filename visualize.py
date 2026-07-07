"""
시각화 도구 모음
================
1. 정규화 전후 진동 신호 비교 (4개 클래스)
2. TST vs LSTM 학습 곡선 비교
3. StepLR 학습률 변화

한글 폰트가 없는 환경에서는 plt.rcParams 부분을 주석 처리하세요.
"""

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# 한글 폰트 (Windows: Malgun Gothic / Mac: AppleGothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = "./data/cwru"
os.makedirs("./results", exist_ok=True)


# ============================================================
# 1. 정규화 전후 진동 신호 비교
# ============================================================
def plot_normalization():
    files = {
        'Normal': os.path.join(DATA_PATH, "Normal_0.mat"),
        'Ball':   os.path.join(DATA_PATH, "B007_0.mat"),
        'IR':     os.path.join(DATA_PATH, "IR007_0.mat"),
        'OR':     os.path.join(DATA_PATH, "OR0076_0.mat"),
    }

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for i, (label, fpath) in enumerate(files.items()):
        mat = scipy.io.loadmat(fpath)
        de_key = [k for k in mat.keys() if 'DE_time' in k][0]
        signal = mat[de_key].flatten()[:1024]
        signal_scaled = (signal - np.mean(signal)) / np.std(signal)

        axes[i][0].plot(signal, color='blue')
        axes[i][0].set_title(f"{label} - Before Normalization")
        axes[i][0].set_ylabel("Amplitude")

        axes[i][1].plot(signal_scaled, color='red')
        axes[i][1].set_title(f"{label} - After Normalization")
        axes[i][1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig("./results/normalization.png", dpi=180)
    plt.show()


# ============================================================
# 2. TST vs LSTM 학습 곡선 비교
# ============================================================
def plot_comparison():
    epochs = list(range(1, 21))

    # 실험 결과 (Val Accuracy)
    tst_val = [0.9487, 0.9278, 0.9594, 0.9476, 0.9594, 0.9600, 0.9684, 0.9549,
               0.9645, 0.9775, 0.9611, 0.9622, 0.9538, 0.9583, 0.9690, 0.9684,
               0.9718, 0.9662, 0.9696, 0.9651]
    lstm_val = [0.4713, 0.3890, 0.4893, 0.3619, 0.4363, 0.5073, 0.3241, 0.4977,
                0.4532, 0.4927, 0.4775, 0.6691, 0.6849, 0.8134, 0.8168, 0.7762,
                0.9256, 0.8174, 0.8997, 0.9064]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, tst_val, 'o-', color='#C00000', linewidth=2,
             markersize=6, label='TST (Transformer)')
    plt.plot(epochs, lstm_val, 's-', color='#1565C0', linewidth=2,
             markersize=6, label='LSTM')

    plt.axhline(y=0.9775, color='#C00000', linestyle='--', alpha=0.4)
    plt.annotate('TST best 97.75% (Epoch 10)', xy=(10, 0.9775), xytext=(11, 0.90),
                 fontsize=10, color='#C00000',
                 arrowprops=dict(arrowstyle='->', color='#C00000'))
    plt.annotate('LSTM final 90.64%', xy=(20, 0.9064), xytext=(14, 0.60),
                 fontsize=10, color='#1565C0',
                 arrowprops=dict(arrowstyle='->', color='#1565C0'))

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('TST vs LSTM Learning Curve (Same Conditions)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.2, 1.05)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig("./results/tst_vs_lstm.png", dpi=180, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. StepLR 학습률 변화
# ============================================================
def plot_lr_schedule():
    epochs = list(range(1, 21))
    lr, current = [], 0.001
    for e in epochs:
        if e > 1 and (e - 1) % 5 == 0:
            current *= 0.5
        lr.append(current)

    plt.figure(figsize=(8, 4))
    plt.step(epochs, lr, where='post', color='#C00000', linewidth=2)
    plt.fill_between(epochs, lr, step='post', alpha=0.2, color='#C00000')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('StepLR (halved every 5 epochs)')
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig("./results/lr_schedule.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    plot_comparison()      # 데이터 없이도 실행 가능 (결과값 하드코딩)
    plot_lr_schedule()     # 데이터 없이도 실행 가능
    # plot_normalization() # .mat 데이터 필요
