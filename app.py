from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import gdown

app = Flask(__name__)

# =========================
# Model Download from Google Drive
# =========================
MODEL_PATH = "brain_tumor.keras"

# Google Drive file ID (from your link)
FILE_ID = "1gx7S8hEBZ47V2hNSDUa1bBmhlnlT4igw"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model safely
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None


# =========================
# Class Labels
# =========================
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


# =========================
# Upload Folder Setup
# =========================
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================
# Check valid file
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================
# Prediction Function
# =========================
def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    probs = prediction[0]
    max_prob = float(np.max(probs))
    predicted_class = int(np.argmax(probs))

    confidence = max_prob * 100

    # =========================
    # Confidence Threshold
    # =========================
    THRESHOLD = 65

    if max_prob < (THRESHOLD / 100):
        return (
            "⚠ This image does not match trained brain MRI patterns. Please upload a valid MRI scan.",
            confidence
        )

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

        if not file or file.filename == "":
            error = "No file selected"
            return render_template("index.html", error=error)

        if not allowed_file(file.filename):
            error = "Invalid file format. Please upload PNG, JPG, JPEG."
            return render_template("index.html", error=error)

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            prediction, confidence = predict_tumor(filepath)
            image_name = filename

        except Exception as e:
            error = f"Prediction error: {str(e)}"
            print(error)

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
    app.run(host="0.0.0.0", port=10000, debug=False)