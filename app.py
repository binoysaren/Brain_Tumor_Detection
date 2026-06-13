from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os
import gdown

# =========================
# Flask App
# =========================
app = Flask(__name__)

# =========================
# Paths
# =========================
MODEL_PATH = "brain_tumor.keras"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# Google Drive Model Config
# =========================
FILE_ID = "1qOFNhH-VdIlM8VE8O6X81YkEX-pfkDRL"

model = None

# =========================
# Load Model Safely
# =========================
def load_model_safe():
    global model

    try:
        # Download model if not exists
        if not os.path.exists(MODEL_PATH):
            print("⬇ Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

        # Load model
        print("📦 Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")

    except Exception as e:
        print("❌ Model loading failed:", e)
        model = None


load_model_safe()

# =========================
# Class Labels
# =========================
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# =========================
# Helpers
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================
# Prediction Function
# =========================
def predict_tumor(img_path):
    if model is None:
        return "Model not loaded", 0.0

    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

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

        if not allowed_file(file.filename):
            error = "Invalid file type (only png, jpg, jpeg allowed)"
            return render_template("index.html", error=error)

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            

            prediction, confidence = predict_tumor(filepath)
            image_name = filename

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
# Run App (Render Safe)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
