import os
import urllib.request

# Must be set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1xJoKVqimMWcOOM1QTF8VIXXPrk1KYGs3"


def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _ = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)
    print("Model loaded.")
    return model