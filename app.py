import joblib
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= APP =================
app = Flask(__name__)

# ================= LOAD MODEL =================
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ================= CNN =================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224,224,3)
)

# ================= TEXTURE =================
def get_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = cv2.Laplacian(gray, cv2.CV_64F)
    hist = np.histogram(lbp.ravel(), bins=32, range=(-255,255))[0]
    return hist / (np.sum(hist) + 1e-7)

def get_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0,1,2],
        None,
        [8,8,8],
        [0,180,0,256,0,256]
    ).flatten()
    return hist / (np.sum(hist) + 1e-7)

# ================= PREPROCESS =================
def preprocess_image(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224,224))
    return img

# ================= FEATURE EXTRACTION =================
def extract_features(img):
    x = preprocess_input(img)
    x = np.expand_dims(x, axis=0)

    cnn_feat = base_model.predict(x, verbose=0).flatten()

    lbp_feat = get_lbp(img)
    hsv_feat = get_hsv(img)

    return np.concatenate([cnn_feat, lbp_feat, hsv_feat])

# ================= LABEL MAP =================
label_map = {
    "healthy": "Healthy (Sehat)",
    "leaf curl": "Leaf Curl (Keriting Daun)",
    "leaf spot": "Leaf Spot (Bercak Daun)",
    "whitefly": "Whitefly (Kutu Putih)",
    "yellowish": "Yellowish (Virus Kuning)"
}

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
        # langsung dari memory (NO SAVE FILE)
        img = Image.open(file.stream)

        # preprocess
        img = preprocess_image(img)

        # feature extraction
        features = extract_features(img).reshape(1, -1)

        # prediction
        probs = svm_model.predict_proba(features)[0]

        idx = np.argmax(probs)
        confidence = float(np.max(probs))

        raw_label = label_encoder.inverse_transform([idx])[0].lower()

        # ================= LOGIC =================
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