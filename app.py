import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

from feature_extractor import extract_features

# ================= APP =================
app = Flask(__name__)

# ================= LOAD MODEL =================
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

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
        features = extract_features(file).reshape(1, -1)

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