from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

# =========================
# Flask App
# =========================
app = Flask(__name__)

# =========================
# TensorFlow Safety Fix
# =========================
tf.keras.backend.clear_session()

# =========================
# Load Model Safely
# =========================
try:
    model_path = os.path.join(os.path.dirname(__file__), "brain_tumor.keras")
    model = load_model(model_path, compile=False)
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
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# Prediction Function
# =========================
def predict_tumor(img_path):

    if model is None:
        raise Exception("Model not loaded")

    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    print("Raw prediction:", prediction)

    probs = prediction[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class] * 100)

    if predicted_class < 0 or predicted_class >= len(CLASSES):
        return "Unknown", confidence

    return CLASSES[predicted_class], confidence

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

        if model is None:
            error = "Model not loaded properly"
            return render_template("index.html", error=error)

        file = request.files.get("file")

        if file and file.filename != "":

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            try:
                prediction, confidence = predict_tumor(filepath)
                image_name = file.filename

            except Exception as e:
                error = str(e)
                print("Prediction error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_name=image_name,
        error=error
    )

# =========================
# Run App (Production Ready)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
