
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# =========================
# Load Model Safely
# =========================
try:
    model = load_model("brain_tumor.keras", compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

# =========================
# Class Labels (must match training)
# =========================
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
# CLASSES = ["notumor", "glioma", "meningioma", "pituitary"]
# CLASSES = ["glioma", "pituitary", "notumor", "meningioma"]

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
    img = image.load_img(img_path, target_size=(128, 128))  # FIXED SIZE
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    print("Raw prediction:", prediction)

    predicted_class = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]) * 100)

    # Safety check to avoid list index error
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
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)