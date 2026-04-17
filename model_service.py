import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
from PIL import Image
import joblib

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= LOAD MODEL SEKALI =================
print("🔥 Loading SVM model...")
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("🔥 Loading CNN model...")
cnn_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(160, 160, 3),
    alpha=0.35
)

print("✅ Model ready!")

# ================= FEATURE =================
def get_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = cv2.Laplacian(gray, cv2.CV_64F)
    hist = np.histogram(lbp.ravel(), bins=8, range=(-255,255))[0]
    return hist / (np.sum(hist) + 1e-7)

def get_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[4,4,2],
                        [0,180,0,256,0,256]).flatten()
    return hist / (np.sum(hist) + 1e-7)

# ================= PREDICT =================
def predict_image(file):
    try:
        img = Image.open(file).convert("RGB")
    except:
        return {"error": "File bukan gambar valid"}

    img = img.resize((160,160))
    img = np.array(img)

    # CNN FEATURE (optimized)
    x = preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    cnn_feat = cnn_model(x, training=False).numpy().flatten()

    # TEXTURE
    lbp_feat = get_lbp(img)
    hsv_feat = get_hsv(img)

    features = np.concatenate([cnn_feat, lbp_feat, hsv_feat]).reshape(1, -1)

    probs = svm_model.predict_proba(features)[0]
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = label_encoder.inverse_transform([idx])[0].lower()

    return {
        "label": label,
        "confidence": confidence
    }