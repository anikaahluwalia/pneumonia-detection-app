from flask import Flask, render_template, request
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import traceback

from config import IMG_SIZE, HOST, PORT

app = Flask(__name__)

model = None
model_error = None


def get_model():
    global model, model_error

    if model is not None:
        return model

    if model_error is not None:
        raise RuntimeError(model_error)

    try:
        from model_loader import load_model
        model = load_model()
        return model
    except Exception as e:
        model_error = f"Model failed to load: {str(e)}"
        traceback.print_exc()
        raise RuntimeError(model_error)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            file = request.files.get("file")
            threshold_raw = request.form.get("threshold", "0.5")

            if not file or file.filename == "":
                raise ValueError("Please upload an image file.")

            try:
                threshold = float(threshold_raw)
            except ValueError:
                raise ValueError("Threshold must be a number.")

            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0 and 1.")

            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            img = img.resize(IMG_SIZE)

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            loaded_model = get_model()
            prob = float(loaded_model(img_batch, training=False).numpy()[0][0])
            prediction = "PNEUMONIA" if prob >= threshold else "NORMAL"

            confidence_gap = abs(prob - threshold)
            if confidence_gap > 0.25:
                confidence = "High"
            elif confidence_gap > 0.10:
                confidence = "Medium"
            else:
                confidence = "Low"

            result = {
                "prediction": prediction,
                "p_pneumonia": round(prob, 3),
                "p_normal": round(1 - prob, 3),
                "confidence": confidence,
                "threshold": threshold,
            }

        except UnidentifiedImageError:
            error = "Uploaded file is not a valid image."
        except Exception as e:
            error = str(e)
            traceback.print_exc()

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)