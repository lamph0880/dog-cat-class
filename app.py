import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image
import io

# 앱 제목
st.set_page_config(page_title="고양이/강아지 분류기", page_icon="🐾")
st.title("🐱 고양이 & 🐶 강아지 분류 웹서비스")
st.write("이미지를 업로드하면 인공지능 모델(Xception)이 고양이인지 강아지인지 판별해 줍니다!")

@st.cache_resource
def load_model():
    # 모델 경로 지정
    model_path = "best_model_xception.keras"
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
    st.success("✅ 모델을 성공적으로 불러왔습니다!")
except Exception as e:
    st.error(f"❌ 모델을 불러오는데 실패했습니다: {e}")
    st.stop()

# 파일 업로더
uploaded_file = st.file_uploader("이미지 파일을 선택하세요 (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 이미지 열기
    img = Image.open(uploaded_file)
    st.image(img, caption="업로드된 이미지", use_column_width=True)
    
    st.write("🔍 분석 중...")
    
    # 모델에 맞게 이미지 전처리
    # Xception 모델의 기본 입력 크기는 (299, 299)
    img_resized = img.resize((299, 299))
    
    # 이미지가 RGB가 아닐 경우 변환 (예: 흑백, RGBA)
    if img_resized.mode != "RGB":
        img_resized = img_resized.convert("RGB")
        
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # 배치 차원 추가
    img_array = preprocess_input(img_array) # Xception 전처리
    
    # 예측 수행
    predictions = model.predict(img_array)
    
    # 결과 해석 (일반적으로 Keras ImageDataGenerator는 알파벳 순으로 클래스 인덱스를 부여합니다: 0=Cat, 1=Dog)
    # 모델의 출력 노드 개수에 따라 이진 분류(sigmoid)인지 다중 분류(softmax)인지 판단
    if predictions.shape[-1] == 1:
        # 이진 분류 (출력 1개)
        prob_dog = float(predictions[0][0])
        prob_cat = 1.0 - prob_dog
        
        if prob_dog > 0.5:
            result = "🐶 강아지"
            confidence = prob_dog * 100
        else:
            result = "🐱 고양이"
            confidence = prob_cat * 100
            
    else:
        # 다중 분류 (출력 2개 이상)
        prob_cat = float(predictions[0][0])
        prob_dog = float(predictions[0][1])
        
        if prob_dog > prob_cat:
            result = "🐶 강아지"
            confidence = prob_dog * 100
        else:
            result = "🐱 고양이"
            confidence = prob_cat * 100
            
    # 결과 출력
    st.markdown(f"### 🎯 판별 결과: **{result}** 입니다!")
    st.markdown(f"**확신도 (Confidence):** {confidence:.2f}%")
