import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= APP =================
app = Flask(__name__)

# ================= LOAD MODEL =================
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ================= CNN MODEL (GLOBAL - OPTIMIZED) =================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

# ================= TEXTURE FEATURE (NO CV2 - FIXED) =================
def get_lbp_like(img):
    # simple texture replacement (NO CV2)
    img = np.array(img.convert("L"))
    hist = np.histogram(img.ravel(), bins=32, range=(0, 255))[0]
    return hist / (np.sum(hist) + 1e-7)

def get_hsv_like(img):
    img = np.array(img.convert("HSV"))
    hist = np.histogram(img.ravel(), bins=32, range=(0, 255))[0]
    return hist / (np.sum(hist) + 1e-7)

# ================= PREPROCESS =================
def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    return img

# ================= FEATURE EXTRACTION =================
def extract_features(img):
    img_np = np.array(img)

    # CNN feature
    x = preprocess_input(img_np)
    x = np.expand_dims(x, axis=0)
    cnn_feat = base_model.predict(x, verbose=0).flatten()

    # texture features
    lbp_feat = get_lbp_like(img)
    hsv_feat = get_hsv_like(img)

    return np.concatenate([cnn_feat, lbp_feat, hsv_feat])

# ================= LABEL MAP =================
label_map = {
    "healthy": "Healthy (Sehat)",
    "leaf curl": "Leaf Curl (Keriting Daun)",
    "leaf spot": "Leaf Spot (Bercak Daun)",
    "whitefly": "Whitefly (Kutu Putih)",
    "yellowish": "Yellowish (Virus Kuning)"
}

# ================= ROUTE =================
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
        img = preprocess_image(file)

        features = extract_features(img).reshape(1, -1)

        probs = svm_model.predict_proba(features)[0]
        idx = int(np.argmax(probs))
        confidence = float(np.max(probs))

        raw_label = label_encoder.inverse_transform([idx])[0].lower()

        if confidence < 0.60:
            result = "Tidak yakin / coba gambar lain"
        else:
            result = label_map.get(raw_label, raw_label)

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)