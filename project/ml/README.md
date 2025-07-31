# 📄 한국어 감정 분석기 (Korean Sentiment Analyzer)

한국어 문장을 입력받아 감정을 분류하는 경량 머신러닝 기반 감정 분석기입니다.

## 🎯 프로젝트 개요

| 항목             | 내용                                                |
| ---------------- | --------------------------------------------------- |
| 📌 **주제**      | 한국어 문장을 입력하면 감정을 분류하는 감정 분석기  |
| 🧠 **목적**      | 머신러닝 기초 실습, 경량 NLP 모델 구현, 성능 비교   |
| 🧪 **감정 라벨** | 우울 / 불안 / 정상 (3-class 분류)                   |
| 💬 **입력**      | 한 문장 또는 텍스트 (한국어)                        |
| 🖥 **출력**       | 예측 감정 라벨 및 신뢰도                            |
| ⚡ **특징**      | 형태소 분석 없이 단순 split + TF-IDF 기반 경량 모델 |

## 🧱 프로젝트 구조

```
sentiment-analyzer/
│
├── data/                        # 데이터 폴더
│   ├── raw/                     # 원본 CSV 데이터 (AIHub 등에서 받은 것)
│   └── processed/               # 전처리된 CSV 저장 (cleaned, labeled)
│
├── notebooks/                   # Jupyter 노트북 작업 공간
│   ├── 01_EDA.ipynb             # 데이터 탐색 및 시각화
│   ├── 02_Preprocessing.ipynb   # 텍스트 정제 및 전처리
│   ├── 03_Modeling.ipynb        # TF-IDF + ML 모델 학습 및 평가
│   └── 04_Enhanced_Training.ipynb # 데이터 증강 및 향상된 모델 훈련
│
├── models/                      # 학습된 모델 저장 (Pickle)
│   ├── best_model.pkl           # 최고 성능 모델 (LogReg/SVM/RF)
│   ├── tfidf_vectorizer.pkl     # TF-IDF 벡터라이저
│   ├── enhanced_model.pkl       # 데이터 증강 모델 (백업)
│   └── enhanced_tfidf_vectorizer.pkl # 증강 모델용 벡터라이저
│
├── app/                         # Streamlit 웹 앱
│   └── app.py                   # 사용자 입력 → 감정 예측 웹 인터페이스
│
├── requirements.txt             # 필요한 패키지 목록 (경량화)
├── .gitignore                   # Git 무시 파일 목록
├── run.sh                       # 웹 앱 실행 스크립트 (옵션)
└── README.md                    # 프로젝트 설명 (이 파일)
```

### 📦 구성 설명

| 폴더/파일          | 설명                                                   |
| ------------------ | ------------------------------------------------------ |
| `data/raw/`        | 원본 CSV, JSON 파일 등 다운로드된 데이터 저장          |
| `data/processed/`  | 전처리된 문장, 라벨 포함된 CSV 저장                    |
| `notebooks/`       | EDA, 전처리, 모델링을 단계별로 분리해서 실험 가능      |
| `models/`          | 학습된 모델과 벡터라이저 저장 → 웹앱에서 불러와서 예측 |
| `app/`             | Streamlit 기반 감정 분석기 프론트엔드                  |
| `requirements.txt` | pip install로 환경 셋업 (경량화된 패키지만)            |
| `README.md`        | 프로젝트 설명서 → 발표 및 공유용                       |
| `run.sh`           | 웹 앱 실행 자동화 스크립트 (선택사항)                  |

## 🔧 환경 설정

### 시스템 요구사항

- Python 3.8.x 이상
- ⚡ **경량화**: Java나 별도 형태소 분석기 불필요

### 설치 방법

1. **저장소 클론**

```bash
git clone <repository-url>
cd sentiment-analyzer
```

2. **가상환경 생성 및 활성화**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

3. **패키지 설치**

```bash
pip install -r requirements.txt
```

### 필요 라이브러리 (requirements.txt)

```txt
# 데이터 처리
pandas==1.5.3
numpy==1.24.4

# 머신러닝 모델
scikit-learn==1.2.2
joblib==1.5.1

# 시각화 (EDA 및 평가용)
matplotlib==3.6.3
seaborn==0.12.2

# 웹 앱 (Streamlit 기반 실시간 감정 분석)
streamlit==1.33.0

# 한글 워드클라우드 (선택 – 감정 단어 시각화용)
wordcloud==1.9.3
```

## 🧪 데이터셋

| 항목     | 설명                                        |
| -------- | ------------------------------------------- |
| **출처** | AIHub 감성대화 말뭉치 (공식 데이터셋)       |
| **라벨** | 우울(0), 불안(1), 정상(2) - 숫자로 매핑     |
| **수량** | 약 10,000~50,000문장 (다운로드 버전에 따라) |
| **포맷** | CSV 형식 → Pandas DataFrame으로 자동 처리   |
| **특징** | 실제 대화 데이터 기반, 전문적 라벨링        |

## ⚙️ 모델링 파이프라인

### 1. 데이터 전처리 (경량화)

- **텍스트 정제**: 특수문자 제거, 공백 정리
- **단순 토큰화**: 공백 기반 split() 사용
- **불용어 제거**: 기본 한국어 불용어 (조사, 어미 등)
- **소문자 변환**: 영어 단어 정규화

### 2. 특성 추출

- **TF-IDF 벡터화**: 단어 빈도-역문서 빈도
- **N-gram 범위**: (1,2) - 단어 및 2단어 조합
- **최대 특성 수**: 5,000개 (경량화)
- **최소/최대 문서 빈도**: min_df=2, max_df=0.9

### 3. 모델 학습

| 모델                             | 설명             | 장점                                 |
| -------------------------------- | ---------------- | ------------------------------------ |
| **Logistic Regression**          | 로지스틱 회귀    | 해석 가능성, 빠른 추론, 안정적 성능  |
| **SVM (Support Vector Machine)** | 서포트 벡터 머신 | 텍스트 분류에 우수한 성능, 빠른 학습 |
| **Random Forest**                | 랜덤 포레스트    | 앙상블 기법, 과적합 방지             |

**모델 선택**: 3개 모델을 비교하여 최고 성능 모델을 자동 선택하고 저장

### 4. 모델 평가

- **정확도 (Accuracy)**
- **F1-Score (Macro/Micro Average)**
- **혼동 행렬 (Confusion Matrix)**
- **정밀도/재현율 (Precision/Recall)**

### 5. 시각화

- 감정별 단어 빈도 분석
- TF-IDF 중요 특성 시각화
- 모델 성능 비교 차트
- 워드클라우드 (감정별)

## 🚀 사용 방법

### 1. 데이터 준비 및 EDA

**AIHub 데이터 다운로드 후:**

```bash
# AIHub에서 다운로드한 파일을 data/raw/ 폴더에 배치
# 파일명을 korean_sentiment_dataset.csv로 변경

jupyter notebook notebooks/01_EDA.ipynb
```

**⚠️ 참고**: AIHub 데이터의 컬럼명이 다를 경우, 전처리 노트북에서 자동 매핑됩니다.

### 2. 텍스트 전처리

```bash
jupyter notebook notebooks/02_Preprocessing.ipynb
```

### 3. 모델 학습 및 평가

```bash
jupyter notebook notebooks/03_Modeling.ipynb
```

### 4. 데이터 증강 및 향상된 모델 (선택사항)

```bash
jupyter notebook notebooks/04_Enhanced_Training.ipynb
```

### 5. 웹 앱 실행

```bash
streamlit run app/app.py
```

또는 실행 스크립트 사용:

```bash
chmod +x run.sh
./run.sh
```

## 💻 웹 데모 기능

Streamlit 기반 경량 감정 분석 웹 애플리케이션:

- 📝 **텍스트 입력**: 한국어 문장 직접 입력
- 🎯 **즉시 감정 예측**: 우울/불안/정상 분류 결과
- 📊 **신뢰도 표시**: 각 감정별 확률 점수 시각화
- 🧠 **스마트 규칙**: 키워드 매칭 기반 보완 분류
- ⚡ **빠른 처리**: 형태소 분석 없이 즉시 결과 제공
- 📈 **모델 성능**: 학습된 모델의 평가 지표 확인
- 🔄 **자동 모델 선택**: enhanced → basic 모델 자동 폴백

## 📊 예상 결과

- **현재 달성**: 95%+ 정확도 (AIHub 데이터 기준)
- **F1-Score**: 0.90 이상 (균형잡힌 성능)
- **처리 속도**: 문장당 < 0.5초 (매우 빠름)
- **모델 크기**: < 50MB (경량화)
- **안정성**: 다중 모델 비교로 최적 선택

## 🔮 향후 개선 계획

- [ ] **딥러닝 모델 적용**: 사전 훈련된 BERT 모델 활용
- [ ] **감정 세분화**: 기쁨, 슬픔, 분노, 놀람 등 추가
- [ ] **실시간 분석**: FastAPI 기반 REST API 서버 구축
- [ ] **모바일 최적화**: 경량 모델로 모바일 앱 지원
- [ ] **다국어 지원**: 영어, 일본어 등 확장
- [ ] **데이터 증강**: 역번역, 동의어 치환 등으로 데이터 확장

## 💡 경량화 장점

- ✅ **빠른 설치**: Java나 복잡한 의존성 없음
- ✅ **빠른 실행**: 형태소 분석 과정 생략으로 속도 향상
- ✅ **작은 용량**: 모델 파일 크기 최소화
- ✅ **쉬운 배포**: 클라우드 환경에서 빠른 배포 가능
- ✅ **낮은 메모리**: 리소스 사용량 최소화

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 개발자

- **이름**: [Your Name]
- **이메일**: [your.email@example.com]
- **GitHub**: [github.com/yourusername]

## 📚 참고 자료

- [Scikit-learn 문서](https://scikit-learn.org/) - TF-IDF 및 SVM 사용법
- [AIHub 감성대화 말뭉치](https://aihub.or.kr/) - 학습 데이터셋
- [Streamlit 문서](https://docs.streamlit.io/) - 웹 앱 개발
- [Pandas 문서](https://pandas.pydata.org/) - 데이터 전처리
- [Matplotlib/Seaborn](https://matplotlib.org/) - 데이터 시각화

---

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!
