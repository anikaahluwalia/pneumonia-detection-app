import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/uc?export=download&id=1xJoKVqimMWcOOM1QTF8VIXXPrk1KYGs3",
)

# Inference
IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.5

# Server
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 5051))
