import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

# Load model
def load_model():
    return tf.keras.models.load_model('plant_disease_cnn_model.keras')

model = load_model()

# Image preprocessing
def model_predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Class labels
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
               'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']

# Streamlit UI
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

st.sidebar.title('🌿 Plant Disease Detection System')
app_mode = st.sidebar.radio('Navigation', ['Home', 'Disease Recognition'])

# Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.image('plantdetect.jpg', use_container_width=True)
    st.write("\n### 🔬 AI-Powered Plant Disease Diagnosis")
    st.write("This application uses deep learning to identify plant diseases from images, helping farmers and researchers maintain healthy crops.")

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.markdown("## 📷 Upload an Image for Diagnosis")
    uploaded_file = st.file_uploader("📂 Drag and drop file here or click to browse", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
            save_path = os.path.join(os.getcwd(), uploaded_file.name)
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ Image Uploaded Successfully")
        
        with col2:
            if st.button("🔍 Predict Disease"):
                result_index = model_predict(save_path)
                st.success(f"🌿 Model Prediction: **{class_names[result_index]}**")
