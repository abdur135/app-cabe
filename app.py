import os
import joblib
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

# ================= APP =================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODEL =================
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ================= CNN FEATURE EXTRACTOR =================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224,224,3)
)

# ================= LABEL =================
label_mapping = {
    "healthy": "Healthy (Sehat)",
    "leaf curl": "Leaf Curl (Keriting Daun)",
    "leaf spot": "Leaf Spot (Bercak Daun)",
    "whitefly": "Whitefly (Kutu Putih)",
    "yellowish": "Yellowish (Virus Kuning)"
}

# ================= RESIZE KEEP RATIO =================
def resize_keep_ratio(img):
    h, w, _ = img.shape
    scale = 224 / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    img = cv2.resize(img, (new_w, new_h))

    top = (224 - new_h) // 2
    bottom = 224 - new_h - top
    left = (224 - new_w) // 2
    right = 224 - new_w - left

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0,0,0]
    )
    return img

# ================= PREPROCESS (WAJIB SAMA TRAINING) =================
def process_image(img):
    img = img.convert("RGB")
    img = np.array(img)

    h, w, _ = img.shape
    if h < 100 or w < 100:
        raise ValueError("Gambar terlalu kecil")

    # resize saja (NO SHARPEN / NO COLOR SHIFT / NO NOISE)
    img = resize_keep_ratio(img)

    # CNN preprocess
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img

# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    try:
        # ================= SAVE IMAGE =================
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # ================= LOAD IMAGE =================
        img = Image.open(path)
        img_array = process_image(img)

        # ================= FEATURE EXTRACTION =================
        features = base_model.predict(img_array, verbose=0).flatten()

        # ================= SVM PREDICTION =================
        probs = svm_model.predict_proba(features.reshape(1, -1))[0]

        pred_idx = np.argmax(probs)
        confidence = float(np.max(probs))

        raw_label = label_encoder.inverse_transform([pred_idx])[0].lower()

        # ================= RESULT =================
        if confidence < 0.65:
            result = "Tidak yakin / coba gambar lain"
        else:
            result = label_mapping.get(raw_label, raw_label)

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)