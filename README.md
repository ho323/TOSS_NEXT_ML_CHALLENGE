# TOSS NEXT ML CHALLENGE

팀명: 도지코인  
대회 링크: https://dacon.io/competitions/official/236575/overview/description  

<img width="1877" height="382" alt="image" src="https://github.com/user-attachments/assets/18644147-6326-4a2e-b444-44efe4b270cb" />  

---

## 📋 프로젝트 개요

TOSS NEXT ML CHALLENGE 대회 참가 프로젝트입니다. 다양한 딥러닝 모델과 앙상블 기법을 활용하여 예측 성능을 향상시켰습니다.

## 🏗️ 프로젝트 구조

```
TOSS_NEXT_ML_CHALLENGE/
├── HDCN/                    # Hybrid Deep Cross Network 모델
│   ├── hdcn_train.py       # HDCN 학습 스크립트
│   └── hdcn_inference.py   # HDCN 추론 스크립트
├── Hybrid_GDCN/            # Hybrid Graph Deep Cross Network 모델
│   ├── Hybrid_GDCN_train.ipynb
│   ├── Hybrid_GDCN_inference.ipynb
│   ├── basic_layers.py
│   └── model_hybrid_gdcn_5epch.pt
├── XGB/                     # XGBoost 모델
│   ├── train.ipynb
│   ├── inference.ipynb
│   └── xgb_model.json
├── models_weight/           # 학습된 모델 가중치
│   ├── hdcn_noseq.pth
│   ├── model_hybrid_gdcn_5epch.pt
│   └── xgb_model.json
├── output/                  # 예측 결과 파일
│   ├── hdcn.csv
│   ├── hybrid_gdcn.csv
│   └── xgb_infer.csv
├── base.py                  # 기본 추상 클래스
├── main.ipynb               # 메인 실행 노트북
├── inference.ipynb          # 통합 추론 노트북
├── requirements.txt         # 패키지 의존성
└── README.md
```

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd TOSS_NEXT_ML_CHALLENGE
```

### 2. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

**주의**: PyTorch는 CUDA 버전에 맞게 별도로 설치해야 할 수 있습니다.
```bash
# CUDA 12.1 버전 예시
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## 📦 주요 의존성

- **numpy**: 1.26.4
- **pandas**: 2.3.2
- **scikit-learn**: 1.6.1
- **xgboost**: 3.0.5
- **lightgbm**: 4.6.0
- **catboost**: 1.2.8
- **optuna**: 4.5.0
- **torch**: 2.3.1+cu121

## 🎯 모델 설명

### 1. HDCN (Hybrid Deep Cross Network)
- 딥러닝 기반 교차 네트워크 모델
- 범주형 변수와 수치형 변수를 효과적으로 처리

### 2. Hybrid_GDCN (Hybrid Graph Deep Cross Network)
- 그래프 구조를 활용한 하이브리드 딥러닝 모델
- 교차 네트워크와 그래프 신경망의 결합

### 3. XGBoost
- 그래디언트 부스팅 기반 트리 모델
- 빠른 학습 속도와 높은 성능

## 🔧 사용 방법

### 데이터 준비
- `data/` 디렉토리에 `train.parquet`와 `test.parquet` 파일을 배치하세요.

### 모델 학습

#### HDCN 모델 학습
```bash
cd HDCN
python hdcn_train.py
```

#### Hybrid_GDCN 모델 학습
- `Hybrid_GDCN/Hybrid_GDCN_train.ipynb` 노트북 실행

#### XGBoost 모델 학습
- `XGB/train.ipynb` 노트북 실행

### 추론 실행

#### 개별 모델 추론
```bash
# HDCN 추론
cd HDCN
python hdcn_inference.py

# Hybrid_GDCN 추론
# Hybrid_GDCN/Hybrid_GDCN_inference.ipynb 실행

# XGBoost 추론
# XGB/inference.ipynb 실행
```

#### 통합 추론
- `inference.ipynb` 또는 `main.ipynb` 노트북 실행

## 📊 모델 구조

<img width="1673" height="929" alt="image" src="https://github.com/user-attachments/assets/e548847f-4b32-4607-9d16-b31323ba7f5b" />  
<img width="1674" height="932" alt="image" src="https://github.com/user-attachments/assets/d8d9a9cf-387a-4b8a-b73d-4a783bb87bde" />

## 📁 출력 파일

모델 추론 결과는 `output/` 디렉토리에 저장됩니다:
- `hdcn.csv`: HDCN 모델 예측 결과
- `hybrid_gdcn.csv`: Hybrid_GDCN 모델 예측 결과
- `xgb_infer.csv`: XGBoost 모델 예측 결과

## ⚙️ 설정

각 모델의 하이퍼파라미터는 해당 스크립트/노트북 내에서 설정할 수 있습니다.

### HDCN 기본 설정
```python
CFG = {
    'BATCH_SIZE': 256,
    'EPOCHS': 5,
    'LEARNING_RATE': 1e-3,
    'SEED': 42
}
```

## 📝 참고사항

- GPU 사용을 권장합니다 (PyTorch 모델 학습 시)
- 데이터 전처리는 각 모델별로 수행됩니다
- 모델 가중치는 `models_weight/` 디렉토리에 저장됩니다

## 👥 팀원

팀명: 도지코인

## 📄 라이선스

이 프로젝트는 TOSS NEXT ML CHALLENGE 대회용으로 작성되었습니다.
