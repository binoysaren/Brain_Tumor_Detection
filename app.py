from flask import Flask, render_template, request
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# Flask App
# =========================
app = Flask(__name__)

# =========================
# TensorFlow Safe Mode
# =========================
tf.get_logger().setLevel('ERROR')

# =========================
# Model Loading
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor.keras")

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

# =========================
# Class Labels (must match training)
# =========================
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# =========================
# Upload Folder
# =========================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# Prediction Function
# =========================
def predict_tumor(img_path):

    if model is None:
        raise Exception("Model is not loaded")

    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    probs = preds[0]
    class_index = int(np.argmax(probs))
    confidence = float(probs[class_index] * 100)

    if class_index < 0 or class_index >= len(CLASSES):
        return "Unknown", confidence

    return CLASSES[class_index], confidence

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_name = None
    error = None

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            error = "No file selected"
            return render_template("index.html", error=error)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        try:
            if model is None:
                raise Exception("Model failed to load")

            prediction, confidence = predict_tumor(filepath)
            image_name = file.filename

        except Exception as e:
            print("Prediction error:", e)
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_name=image_name,
        error=error
    )

# =========================
# Run (Render Ready)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
