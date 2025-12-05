# src/streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
from unet_model import load_unet_model  # import the function

# Load UNet model
@st.cache_resource  # caches the model in Streamlit
def get_model():
    return load_unet_model()

model = get_model()

st.title("Pelvic Fracture Segmentation 🚀")

uploaded_file = st.file_uploader("Upload a pelvic CT/X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image_resized = image.resize((128,128))
    image_array = np.array(image_resized) / 255.0
    image_array = image_array.reshape(1,128,128,1)

    # Model prediction
    pred_mask = model.predict(image_array)[0,:,:,0]
    
    # Display original and mask
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(pred_mask, caption="Predicted Fracture Mask", use_column_width=True)
