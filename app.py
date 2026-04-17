from flask import Flask, request, jsonify, render_template
from model_service import predict_image

# ================= APP =================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # max 2MB upload

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

    result = predict_image(file)

    if "error" in result:
        return jsonify(result)

    label = result["label"]
    confidence = result["confidence"]

    if confidence < 0.6:
        final_result = "Tidak yakin / coba gambar lain"
    else:
        final_result = label_map.get(label, label)

    return jsonify({
        "prediction": final_result,
        "confidence": confidence
    })

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)