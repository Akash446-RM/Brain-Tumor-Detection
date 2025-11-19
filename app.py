# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

@st.cache_resource(show_spinner=False)
def load_my_model(path="BrainTumor10Epochscategorical.h5"):
    model = load_model(path)
    return model

def preprocess_image(image_pil, target_size=(64,64)):
    img = image_pil.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(model, input_tensor):
    pred = model.predict(input_tensor)
    class_id = int(np.argmax(pred, axis=1)[0])
    return pred[0], class_id


# ----------------- UI -----------------
st.title("🧠 Brain Tumor Detector")
st.write("Upload an MRI image (jpg/png). Model: **BrainTumor10Epochscategorical.h5**")

# Load model
with st.spinner("Loading model..."):
    model = load_my_model("BrainTumor10Epochscategorical.h5")

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

    st.subheader("Input image")
    st.image(input_image, use_column_width=True)

    input_tensor = preprocess_image(input_image, target_size=(64,64))

    # Predict
    probs, cls_id = predict(model, input_tensor)

    labels = {0: "NO TUMOR", 1: "TUMOR"}
    predicted_label = labels.get(cls_id, str(cls_id))
    confidence = probs[cls_id]

    st.markdown("### Prediction")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Result", predicted_label)
    with col2:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    st.write("Raw probabilities (no_tumor, yes_tumor):", np.round(probs, 4))

    st.success("Done ✅")
else:
    st.info("Upload an image to get prediction.")

