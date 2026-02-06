import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# =========================
# CẤU HÌNH
# =========================
MODEL_PATH = "human_vs_nonhuman_mobilenetv2.h5"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Human Detection",
    page_icon="🧍",
    layout="centered"
)

st.title(" Human vs Non-Human Detection")
st.write("Upload ảnh để phân loại: **Người / Không phải người**")

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_cnn_model():
    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_cnn_model()

# =========================
# UPLOAD ẢNH
# =========================
uploaded_file = st.file_uploader(
    "Chọn một ảnh",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Hiển thị ảnh
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Ảnh đã upload", use_column_width=True)

    # Tiền xử lý
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # =========================
    # DỰ ĐOÁN
    # =========================
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])  # xác suất là NGƯỜI

    human_percent = prob * 100
    nonhuman_percent = (1 - prob) * 100

    st.subheader(" Kết quả dự đoán")

    st.write(f" **Xác suất NGƯỜI:** {human_percent:.2f}%")
    st.write(f" **Xác suất KHÔNG PHẢI NGƯỜI:** {nonhuman_percent:.2f}%")

    if prob > 0.5:
        st.success(" Dự đoán cuối cùng: **NGƯỜI**")
    else:
        st.warning(" Dự đoán cuối cùng: **KHÔNG PHẢI NGƯỜI**")
