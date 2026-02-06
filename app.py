import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model với cấu hình an toàn cho Keras 2/3
model = tf.keras.models.load_model("human_vs_nonhuman_mobilenetv2.h5", compile=False)

def predict_human(img):
    # Tiền xử lý ảnh giống hệt như khi bạn train
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Dự đoán
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])
    
    if prob > 0.5:
        return f"Dự đoán: NGƯỜI ({prob*100:.2f}%)"
    else:
        return f"Dự đoán: KHÔNG PHẢI NGƯỜI ({(1-prob)*100:.2f}%)"

# Tạo giao diện Gradio
interface = gr.Interface(
    fn=predict_human,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Human Detection App",
    description="Upload ảnh để kiểm tra xem có phải là người hay không."
)

interface.launch()