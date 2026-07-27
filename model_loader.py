import os

# Must be set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from config import MODEL_DIR, MODEL_PATH, MODEL_URL


def _download_model():
    """Download the model weights, handling Google Drive's large-file
    confirmation page correctly (plain urllib fails on this silently)."""
    import gdown

    print("Downloading model...")
    gdown.download(url=MODEL_URL, output=MODEL_PATH, quiet=False, fuzzy=True)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10_000:
        # A valid Keras model file should be well over 10KB. If it's tiny,
        # gdown most likely saved an HTML error/warning page instead.
        raise RuntimeError(
            "Model download failed or returned an invalid file "
            "(check that MODEL_URL points to a public, direct Google Drive file id)."
        )


def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        _download_model()

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _ = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)
    print("Model loaded.")
    return model