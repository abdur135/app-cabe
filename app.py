import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

app = Flask(__name__)

# ================= LOAD =================
svm_model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# LAZY LOAD CNN
base_model = None

def get_model():
    global base_model
    if base_model is None:
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return base_model

# ================= LABEL =================
label_mapping = {
    'healthy': 'Healthy (Sehat)',
    'leaf curl': 'Leaf Curl (Keriting Daun)',
    'leaf spot': 'Leaf Spot (Bercak Daun)',
    'whitefly': 'Whitefly (Kutu Putih)',
    'yellowish': 'Yellowish (Virus Kuning)'
}

# ================= PREPROCESS =================
def process_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)

    if img_array.mean() < 30:
        raise ValueError("Gambar terlalu gelap")

    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ================= ROUTE =================
@app.route('/')
def home():
    return render_template('index.html')

# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file'})

    file = request.files['file']

    try:
        img = Image.open(file.stream)
        img_array = process_image(img)

        model_cnn = get_model()
        features = model_cnn.predict(img_array)
        features = features.flatten().reshape(1, -1)

        probs = svm_model.predict_proba(features)[0]
        pred_class = np.argmax(probs)
        confidence = float(np.max(probs))

        raw_label = label_encoder.inverse_transform([pred_class])[0].lower()

        if confidence < 0.55:
            label = "Tidak yakin"
        else:
            label = label_mapping.get(raw_label, raw_label)

        return jsonify({
            'prediction': label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True)