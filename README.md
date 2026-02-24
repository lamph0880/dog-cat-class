# 🐱🐶 고양이 & 강아지 분류 웹 서비스

이 프로젝트는 Xception 딥러닝 모델(`best_model_xception.keras`)을 사용하여 고양이와 강아지 이미지를 분류하는 간단한 웹 서비스입니다. [Streamlit](https://streamlit.io/)을 활용하여 빠르고 직관적인 UI를 통해 사용자가 사진을 업로드하면 인공지능이 이미지를 분석하고 결과를 예측합니다.

## ✨ 기능
- 이미지 업로드 기능 (`.jpg`, `.png`, `.jpeg` 지원)
- Xception 모델 기반 추론 및 분류
- 강아지/고양이 예측 결과 및 확신도(Confidence) 표시

## 🛠 사용된 기술 (Tech Stack)
- **UI Framework:** Streamlit
- **Deep Learning Model:** TensorFlow / Keras (Xception)
- **Image Processing:** Python Imaging Library (Pillow), NumPy

## 🚀 설치 및 실행 방법

### 1. 저장소 클론 (Clone Repository)
```bash
git clone https://github.com/lamph0880/dog-cat-class.git
cd dog-cat-class
```

### 2. 패키지 설치 (Install Requirements)
해당 프로젝트를 구동하기 위한 패키지들을 설치합니다.
```bash
pip install -r requirements.txt
```

### 3. 웹 서비스 실행 (Run Streamlit App)
웹 서비스를 실행하면 로컬 브라우저가 자동으로 열립니다.
```bash
streamlit run app.py
```

## 📁 모델 파일 (보안 및 용량 이슈)
`best_model_xception.keras` 파일은 너무 크거나 보안상의 이유로 `.gitignore`에 의하여 깃허브 저장소에 포함되어 있지 않습니다.
앱을 정상적으로 동작시키려면 사전에 학습된 `best_model_xception.keras` 파일을 프로젝트 최상위 경로(app.py와 같은 폴더)에 배치해주셔야 합니다.
