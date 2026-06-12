from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import traceback

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model safely
try:
    model = load_model("brain_tumor.keras", compile=False)
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    model = None

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def predict_tumor(img_path):
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        class_index = np.argmax(preds[0])
        confidence = float(np.max(preds[0]) * 100)

        if class_index >= len(CLASSES):
            return "Unknown", confidence

        return CLASSES[class_index], confidence

    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return "Error", 0.0


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    image_name = None
    error = None

    if request.method == "POST":
        try:
            file = request.files.get("file")

            if not file or file.filename == "":
                error = "No file selected"
                return render_template("index.html", error=error)

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            if model is None:
                error = "Model failed to load"
                return render_template("index.html", error=error)

            prediction, confidence = predict_tumor(filepath)
            image_name = file.filename

        except Exception as e:
            error = str(e)
            print("ERROR:", e)
            traceback.print_exc()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_name=image_name,
        error=error
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
