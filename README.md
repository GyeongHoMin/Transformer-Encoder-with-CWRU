# 트랜스포머 인코더 기반 베어링 이상진단

원시 진동 신호로부터 4종 베어링 결함을 분류하는 Time Series Transformer (TST) 모델. CWRU 데이터셋 사용.

**작성자:** 민경호, 한경국립대학교 기계공학과

---

## 개요

수동 특징 추출(feature engineering) 없이 원시 진동 신호에서 직접 베어링 상태 4종을 분류하는 Transformer Encoder 모델입니다. Self-Attention으로 진동 시퀀스의 장거리 의존성을 포착하며, 동일 조건에서 기존 LSTM 대비 우수한 성능을 보입니다.

| 클래스 | 설명 |
|--------|------|
| Normal (N) | 정상 베어링 |
| Ball (B) | 볼 결함 |
| Inner Race (IR) | 내륜 결함 |
| Outer Race (OR) | 외륜 결함 |

---

## 결과

| 지표 | TST | LSTM |
|------|-----|------|
| 테스트 정확도 | **96%** | — |
| Macro F1-score | **0.96** | — |
| 최고 검증 정확도 | **97.75%** (Epoch 10) | 90.64% (Epoch 20) |
| 수렴 속도 | 약 1 epoch | 약 14 epoch |

클래스별 F1: Normal 1.00 / Ball 0.92 / IR 0.96 / OR 0.95

혼동행렬 분석 결과 Normal은 503개 전부 정확히 분류(완벽 구분)되었으며, 결함 3종(Ball/IR/OR) 간에는 진동 신호가 상대적으로 유사하여 일부 혼동이 발생했습니다.

---

## 전체 파이프라인

```
원시 진동 신호 (.mat, 40개 파일)
   ↓  슬라이딩 윈도우 (크기 1024, stride 512, 50% 겹침) → 11,832개 윈도우
   ↓  Z-score 정규화 (StandardScaler)
   ↓  텐서 변환 + Train/Val/Test 분할 (70/15/15)
   ↓  DataLoader (batch_size=64, shuffle)
   ↓  input_proj: Linear(1 → 64)  [1차원 → 64차원 확장]
   ↓  위치 임베딩(positional embedding) 추가
   ↓  Transformer Encoder × 3 (Multi-Head Attention + FFN)
   ↓  Global Average Pooling (GAP)
   ↓  분류 헤드: Linear(64 → 4)
결함 진단 결과
```

---

## 모델 설정

| 하이퍼파라미터 | 값 | 근거 |
|----------------|-----|------|
| 윈도우 크기 | 1024 | 베어링 약 2.5회전 분량 (물리적 근거) |
| Stride | 512 | 50% 겹침 |
| d_model | 64 | 임베딩 차원 |
| nhead | 8 | 어텐션 헤드 수 (64 ÷ 8 = 8) |
| dim_feedforward | 256 | d_model × 4 (Transformer 관례) |
| num_layers | 3 | 인코더 층 수 |
| dropout | 0.1 | 과적합 방지 |
| optimizer | Adam (lr=1e-3) | StepLR (step=5, gamma=0.5) 병행 |
| epochs | 20 | 배치 크기 64 |

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `cwru_tst.py` | 전체 파이프라인: 데이터 로딩, 전처리, TST 모델, 학습, 평가 |
| `lstm_baseline.py` | 성능 비교용 LSTM 모델 |

---

## 실행 방법

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn

# cwru_tst.py에서 데이터 경로 설정 후 실행:
python cwru_tst.py
```

**데이터셋:** [CWRU Bearing Dataset (Kaggle)](https://www.kaggle.com/datasets/astrollama/cwru-case-western-reserve-university-dataset)

결함 신호가 가장 뚜렷하게 나타나는 Drive End(DE) 채널만 사용합니다.

---

## 주요 설계 선택

- **Encoder-only 구조** — 분류는 시퀀스 생성이 불필요하므로 Decoder를 사용하지 않음 (BERT와 동일 방식).
- **CLS 토큰 대신 Global Average Pooling** — 구현이 간단하면서도 유사한 정확도 달성.
- **이상치 제거 미적용** — 진동 신호에서 큰 진폭값은 노이즈가 아닌 결함의 핵심 정보이기 때문.
- **무작위 분할(random split)** — 시계열 예측이 아닌 분류 문제이고, 클래스가 균형(각 24~28%)적이므로 적절함.

---

## 향후 과제

- **Early Stopping** — 최적 시점 자동 선택 (검증 정확도가 Epoch 10에서 최고, 이후 경미한 과적합).
- **부하 조건별 일반화 검증** — 특정 부하(0/1/2 hp)로 학습 후 학습하지 않은 부하(3 hp)로 테스트.
- **파일 단위 분할** — 겹친 윈도우로 인한 약한 정보 공유 제거.
- **하이퍼파라미터 튜닝** — d_model, num_layers 등에 대한 그리드 서치.


