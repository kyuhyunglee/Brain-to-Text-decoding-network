# Brain-to-Text Decoding Network

손글씨 상상(attempted handwriting) 시 운동피질에서 기록된 신경 신호를 텍스트로 디코딩하는 프로젝트입니다.
Willett et al. (2021, *Nature*) "High-performance brain-to-text communication via handwriting"의 RNN 파이프라인을 Google Colab에서 재현하고, 이를 **Transformer 기반 디코더**로 대체·확장하는 실험을 담고 있습니다.

| 노트북 | 내용 | Colab |
|---|---|---|
| `BCIMainController.ipynb` | 원 논문 RNN 파이프라인(Step 1–5) 재현 + 베이스라인 파라미터/지연시간 측정 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyuhyunglee/Brain-to-Text-decoding-network/blob/main/BCIMainController.ipynb) |
| `BCITransformer.ipynb` | Patch 기반 Neural Transformer 디코더 (PyTorch) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyuhyunglee/Brain-to-Text-decoding-network/blob/main/BCITransformer.ipynb) |
| `BCIAdaptiveTransformer.ipynb` | Adaptive Bilinear Attention + g-guided Test-Time Adaptation (실험 단계) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyuhyunglee/Brain-to-Text-decoding-network/blob/main/BCIAdaptiveTransformer.ipynb) |

---

## 데이터

- **출처**: Willett et al. 2021 공개 데이터셋 (Dryad, `doi:10.5061/dryad.wh70rxwmv`)
- **피험자**: T5, 192채널 (Utah array ×2), 10 ms bin 단위 threshold-crossing feature
- **사용 세션**: `t5.2019.05.08`, `t5.2019.11.25`, `t5.2019.12.09`, `t5.2019.12.11`, `t5.2019.12.18`, `t5.2019.12.20`, `t5.2020.01.06`, `t5.2020.01.08`
- **분할**: `HeldOutTrials` 파티션 (`trainTestPartitions_HeldOutTrials.mat`)
- **출력 클래스**: 31개 문자(`a–z`, `>`(공백), `,`, `'`, `~`(마침표), `?`) + 1개 문자 시작(START) 신호 = 32채널

데이터는 Google Drive에 압축 해제한 뒤 노트북의 `rootDir`를 해당 경로로 맞춰 사용합니다.

```
handwritingBCIData/
├── Datasets/<session>/sentences.mat
└── RNNTrainingSteps/
    ├── trainTestPartitions_HeldOutTrials.mat
    ├── Step1_TimeWarping/
    ├── Step2_HMMLabels/HeldOutTrials/<session>_timeSeriesLabels.mat
    ├── Step3_SyntheticSentences/
    └── Step4_RNNTraining/
```

> Transformer 노트북들은 원 파이프라인의 **Step 2 HMM 라벨**을 그대로 사용하므로, `BCIMainController.ipynb`의 Step 2까지는 먼저 실행해야 합니다.

---

## 1. 베이스라인 재현 — `BCIMainController.ipynb`

원 저자 코드([handwritingBCI](https://github.com/fwillett/handwritingBCI))는 TensorFlow 1.15 / Python 3.7 기반이라 최신 Colab(Python ≥3.8)에서 바로 실행되지 않습니다. 이를 해결하기 위해 다음과 같이 환경을 구성했습니다.

- `condacolab`으로 별도 conda 환경(`bci`, Python 3.7, numpy 1.17, scipy 1.1.0, scikit-learn 0.20) 생성
- `tensorflow-gpu==1.15`, CUDA 10.0 / cuDNN 7.6 설치, `protobuf==3.20.3` 고정
- `twpca`는 TF 버전 충돌을 피하기 위해 `--no-deps`로 설치
- 원 노트북들을 `jupyter nbconvert`로 `.py` 변환 후 `bci` 환경 파이썬으로 실행
- Step 4는 Colab에서 백그라운드 학습/실시간 시각화가 동작하지 않아 `%%writefile`로 스크립트를 수정(TF2 behavior 비활성화, 체크포인트 재개 로직 추가)

| Step | 내용 |
|---|---|
| 1 | Time-warped PCA로 단일 문자 trial 정렬 |
| 2 | HMM으로 문장 데이터에 문자 시점 라벨링 |
| 3 | 단일 문자 조각을 이어 붙여 합성 문장 생성 (데이터 증강) |
| 4 | 2-layer GRU (512 units) 학습, 100k step |
| 5 | 추론 및 CER/WER 계산 |

---

## 2. Neural Transformer — `BCITransformer.ipynb`

RNN을 시간축 patch 기반 Transformer Encoder로 대체한 모델입니다.

**구조**
- **Patch Embedding**: `(B, T, 192) → (B, T/P, dim)`. `P=20` bin(200 ms)을 하나의 토큰으로 묶고 LayerNorm → Linear → LayerNorm
- **Positional Embedding**: 학습 가능한 절대 위치 임베딩
- **Encoder**: Pre-Norm Transformer, `dim=384`, `depth=6`, `heads=6`, `mlp_dim=1536`, GELU
- **두 개의 출력 헤드**
  - `char_head` (31-way): 각 시점의 문자 분류 → CrossEntropy (label smoothing 0.1)
  - `start_head` (1-dim): 문자 시작 신호 회귀 → MSE (가중치 ×10)

**데이터 증강**
- **Time Masking**: 문장당 최대 20개 구간(길이 ≤ 7.5%)을 0으로 마스킹 (Time-Masked Transformer 논문 참고)
- **합성 문장**: train split에서 문자별 snippet 라이브러리를 만들고, 무작위로 이어 붙여 합성 문장 생성
  - 속도 변형(0.8–1.2배 리샘플링), white noise, 채널별 constant offset drift
  - 신호 대비 라벨을 100–150 ms 지연시켜 신경 반응 지연을 모사
- 학습셋 = 실제 문장 ×10 + 합성 문장 20,480개

**학습 설정**: AdamW (lr 1e-3, weight decay 0.05), ReduceLROnPlateau, AMP, Google Drive 체크포인트 자동 재개

**디코딩**: START 신호의 sigmoid 확률에서 threshold를 넘는 local peak를 찾고, 그 시점의 argmax 문자를 출력하는 greedy decoding. 같은 문자가 짧은 간격으로 반복될 때만 억제하는 refractory 규칙 적용. 언어 모델은 사용하지 않습니다.

---

## 3. Adaptive Transformer — `BCIAdaptiveTransformer.ipynb` (실험 단계)

세션 간 신호 변화(non-stationarity)에 대응하기 위한 확장 실험입니다.

- **Adaptive Bilinear Attention**: 표준 dot-product score `I = QKᵀ`와 저랭크 bilinear score `QWKᵀ` (`W = U₁U₂`, rank 8)를 게이트 `g`로 혼합
  `scores = g · I + (1 − g) · QWKᵀ`
  - `g`는 표준 어텐션 분포의 엔트로피로부터 계산 (엔트로피가 낮을수록, 즉 확신이 클수록 `g → 1`)
  - 학습 초기에 bilinear 경로로 붕괴하는 것을 막기 위해 epoch에 따라 감소하는 하한선(0.5 → 0.01) 적용
- **DropPath** (stochastic depth, 최대 0.2)
- **g-guided Test-Time Adaptation**: 마지막 레이어의 `g`가 임계값(0.65)보다 높은 테스트 샘플에 한해 entropy minimization으로 patch embedding만 업데이트
- 레이어·헤드별 `g` 분포 시각화(`analyze_confidence`)

---

## 결과

| 모델 | 파라미터 | 추론 지연 (B=1, 12,000 bins) | CER | WER |
|---|---:|---:|---:|---:|
| GRU 베이스라인 (Step 5, TF1) | 2.68 M | 399.5 ms ¹ | 2.84 % | 13.94 % |
| Neural Transformer | 12.37 M | 2.42 ms | 10.63 % | 44.92 % |
| Adaptive Transformer | 11.89 M | 8.67 ms | 80.39 % | 99.49 % |

¹ 동일 구조를 PyTorch로 재구현해 측정한 값

**해석 시 주의사항**
- 베이스라인 CER/WER은 원 파이프라인의 held-out 평가셋 기준이고, Transformer 계열은 test 파티션을 무작위 반분한 **41문장**에서 측정했습니다. 평가셋이 달라 수치를 직접 비교할 수는 없습니다.
- 지연시간은 Transformer가 200 ms 패치로 시퀀스 길이를 1/20로 줄인 효과가 큽니다. 다만 Transformer는 전체 문장을 한 번에 보는 비인과적(bidirectional) 구조이고 GRU는 단방향이라, 실시간 스트리밍 조건에서의 비교는 아닙니다.
- Adaptive Transformer는 아직 수렴하지 않았고(val loss 상승), 모든 테스트 샘플의 `g`가 0.31–0.47로 TTA 임계값(0.65)을 넘지 못해 적응이 한 번도 수행되지 않았습니다. 현재 결과는 방법의 유효성을 판단하기 위한 것이 아니라 진행 중인 실험 기록입니다.

---

## 실행 방법

1. 데이터셋을 다운로드해 Google Drive에 압축 해제
2. 원 저자 레포의 `Step1~5` 노트북과 `charSeqRNN.py` 등을 Drive의 프로젝트 폴더(`handwritingBCI-main`)에 배치
3. `BCIMainController.ipynb`를 Colab GPU 런타임에서 순서대로 실행 (conda 설치 셀 실행 후 커널이 재시작됨)
4. Step 2 라벨 생성이 끝나면 `BCITransformer.ipynb` 또는 `BCIAdaptiveTransformer.ipynb` 실행
   - 의존성: `torch`, `einops`, `jiwer`, `scipy`, `scikit-learn`, `matplotlib`
   - Adaptive 모델은 A100/H100(TF32) 환경에서 실행했습니다.

각 노트북 상단의 `rootDir`, `projectDir`, `checkpoint_dir`를 자신의 Drive 경로에 맞게 수정해야 합니다.

---

## 참고 문헌

- Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., & Shenoy, K. V. (2021). High-performance brain-to-text communication via handwriting. *Nature*, 593, 249–254.
- 원 코드: https://github.com/fwillett/handwritingBCI
- twpca: https://github.com/ganguli-lab/twpca
- Transformer 블록 구현은 [vit-pytorch](https://github.com/lucidrains/vit-pytorch)의 구조를 참고했습니다.
