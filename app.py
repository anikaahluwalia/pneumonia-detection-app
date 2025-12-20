from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io

from model_loader import load_model

app = Flask(__name__)
model = load_model()

IMG_SIZE = (224, 224)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files.get("file")
        threshold = float(request.form.get("threshold", 0.5))

        if file:
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
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

            result = {
                "prediction": prediction,
                "p_pneumonia": round(prob, 3),
                "p_normal": round(1 - prob, 3),
                "confidence": confidence,
                "threshold": threshold,
            }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

