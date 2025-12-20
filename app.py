import os
import json
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/pneumonia_model.keras"
MODEL_URL = "https://drive.google.com/file/d/1xJoKVqimMWcOOM1QTF8VIXXPrk1KYGs3/view"
METRICS_PATH = "models/metrics.json"
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file missing: models/pneumonia_model.keras")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    model(tf.zeros((1, 224, 224, 3)))

    return model



@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    else:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "roc_auc": 0.0,
        }

st.set_page_config(
    page_title="Pneumonia Detection (Chest X-ray)",
    layout="wide",
)

st.markdown(
    """
    <h1 style="text-align:center;">🩻 Pneumonia Detection</h1>
    <p style="text-align:center; color: gray;"></p>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "⚠️ For educational use only. This app is NOT a medical device and should not be used for diagnosis."
)

model = load_model()


st.divider()

st.subheader("📤 Upload a Chest X-ray")

uploaded = st.file_uploader(
    "Upload a JPG or PNG chest X-ray image",
    type=["jpg", "jpeg", "png"],
)

threshold = st.slider(
    "Decision threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Lower threshold = more sensitive to pneumonia",
)

if uploaded is not None:

    img = Image.open(uploaded).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)


    prob = float(model.predict(img_batch, verbose=0)[0][0])
    prediction = "PNEUMONIA" if prob >= threshold else "NORMAL"


    confidence_gap = abs(prob - threshold)
    if confidence_gap > 0.25:
        confidence = "High"
    elif confidence_gap > 0.10:
        confidence = "Medium"
    else:
        confidence = "Low"


    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded X-ray", width=350)

    with col2:
        st.subheader("🧠 Prediction Result")
        st.metric("Predicted Class", prediction)
        st.metric("P(Pneumonia)", f"{prob:.3f}")
        st.metric("P(Normal)", f"{1 - prob:.3f}")
        st.metric("Confidence", confidence)

        st.caption(
            "Prediction is based on a probability threshold "
            f"of {threshold:.2f}."
        )

st.divider()

st.caption(
    "Built with TensorFlow/Keras, NumPy, and Streamlit • Educational demo project"
)
