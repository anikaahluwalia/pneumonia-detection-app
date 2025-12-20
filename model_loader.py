import os
import urllib.request
import tensorflow as tf

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")

MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1xJoKVqimMWcOOM1QTF8VIXXPrk1KYGs3"
)

def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)
    model(tf.zeros((1, 224, 224, 3)))  # build graph

    return model
