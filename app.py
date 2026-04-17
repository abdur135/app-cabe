from flask import Flask, request, jsonify, render_template
from model_service import predict_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB

label_map = {
    "healthy": "Healthy (Sehat)",
    "leaf curl": "Leaf Curl (Keriting Daun)",
    "leaf spot": "Leaf Spot (Bercak Daun)",
    "whitefly": "Whitefly (Kutu Putih)",
    "yellowish": "Yellowish (Virus Kuning)"
}

@app.route("/")
def home():
    return render_template("index.html")

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
        final = "Tidak yakin / coba gambar lain"
    else:
        final = label_map.get(label, label)

    return jsonify({
        "prediction": final,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run()