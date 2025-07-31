#!/bin/bash

# 한국어 감정 분석기 웹 앱 실행 스크립트

echo "🎭 한국어 감정 분석기 시작"
echo "=================================="

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 가상환경이 활성화되어 있습니다: $VIRTUAL_ENV"
else
    echo "⚠️  가상환경이 활성화되지 않았습니다."
    echo "다음 명령어로 가상환경을 활성화하세요:"
    echo "source venv/bin/activate"
    echo ""
    read -p "계속 진행하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 필요한 패키지 설치 확인
echo "📦 패키지 의존성 확인 중..."
python -c "import streamlit, pandas, sklearn, matplotlib, seaborn, wordcloud" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ 모든 필요한 패키지가 설치되어 있습니다."
else
    echo "❌ 일부 패키지가 누락되었습니다."
    echo "다음 명령어로 패키지를 설치하세요:"
    echo "pip install -r requirements.txt"
    exit 1
fi

# 모델 파일 존재 확인
if [ -f "models/tfidf_vectorizer.pkl" ] && [ -f "models/svm_model.pkl" ]; then
    echo "✅ 학습된 모델을 찾았습니다."
else
    echo "⚠️  학습된 모델을 찾을 수 없습니다."
    echo "먼저 다음 노트북들을 실행하여 모델을 학습시키세요:"
    echo "1. notebooks/01_EDA.ipynb"
    echo "2. notebooks/02_Preprocessing.ipynb"  
    echo "3. notebooks/03_Modeling.ipynb"
    echo ""
    echo "📚 학습용 데이터 없이 데모 모드로 실행합니다."
fi

# 포트 확인
PORT=${1:-8501}
echo "🌐 포트 $PORT 에서 웹 앱을 시작합니다..."
echo "브라우저에서 http://localhost:$PORT 로 접속하세요."
echo ""
echo "종료하려면 Ctrl+C를 누르세요."
echo "=================================="

# Streamlit 앱 실행
cd app
streamlit run app.py --server.port $PORT
