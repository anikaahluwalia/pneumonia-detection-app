from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

IMG_SIZE = (224, 224)
model = None   # lazy load

@app.route("/", methods=["GET", "POST"])
def index():
    global model
    result = None

    if model is None:
        from model_loader import load_model
        model = load_model()

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
    app.run(debug=False, use_reloader=False)


