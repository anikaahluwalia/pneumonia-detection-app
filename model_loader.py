import os
import re

# Must be set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from config import MODEL_DIR, MODEL_PATH, MODEL_URL


def _extract_drive_id(url_or_id):
    """Accepts either a raw Google Drive file id, or a full Drive URL, and
    returns just the file id."""
    match = re.search(r"[-\w]{25,}", url_or_id)
    return match.group(0) if match else url_or_id


def _download_model():
    """Download the model weights via gdown, using the file id directly.
    This avoids relying on the `fuzzy` argument, which isn't present in
    all gdown versions."""
    import gdown

    file_id = _extract_drive_id(MODEL_URL)
    print(f"Downloading model (drive id: {file_id})...")
    gdown.download(id=file_id, output=MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10_000:
        raise RuntimeError(
            "Model download failed or returned an invalid file "
            "(check that MODEL_URL points to a public Google Drive file)."
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