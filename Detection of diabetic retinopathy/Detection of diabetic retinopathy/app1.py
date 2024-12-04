import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_model():
    model_path ="model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.title("Detection of Diabetic Retinopathy")
st.text("Upload a retinal image for classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_class(image):
    RGBImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(RGBImg, (224, 224))  # Resize to 224x224
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    return prediction

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    try:
        result = predict_class(image)
        if result[0][0] > 0.5:
            st.write("Diabetic Retinopathy Detected.")
        else:
            st.write("Diabetic Retinopathy Not Detected")
    except Exception as e:
        st.write(f"Error: {str(e)}")
