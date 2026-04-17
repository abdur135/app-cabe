import cv2
import numpy as np
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= CNN MODEL =================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

# ================= TEXTURE FEATURES =================
def get_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = cv2.Laplacian(gray, cv2.CV_64F)
    hist = np.histogram(lbp.ravel(), bins=32, range=(-255,255))[0]
    return hist / (np.sum(hist) + 1e-7)

def get_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256]).flatten()
    return hist / (np.sum(hist) + 1e-7)

# ================= FEATURE EXTRACTION =================
def extract_features(file):
    img = Image.open(file).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224,224))

    x = preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    cnn_feat = base_model.predict(x, verbose=0).flatten()

    lbp_feat = get_lbp(img)
    hsv_feat = get_hsv(img)

    return np.concatenate([cnn_feat, lbp_feat, hsv_feat])