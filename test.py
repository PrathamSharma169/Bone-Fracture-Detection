import os
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import io

IMAGE_SIZE = (224, 224)

def preprocess_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

model = tf.keras.models.load_model('model.h5')
le = joblib.load('label_encoder.pkl')

# ✅ Use os.path.join for safety
img_path = os.path.join("Bone Break Classification", "Fracture Dislocation", "Train", "type20i-lateral_jpg.rf.117a1f8229d0dc4d7970c07ad47c2cc1.jpg")

if not os.path.exists(img_path):
    print("Image not found:", img_path)
else:
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    img_processed = preprocess_image_from_bytes(img_bytes)
    pred = model.predict(img_processed)
    pred_class = np.argmax(pred, axis=1)[0]
    fracture_type = le.inverse_transform([pred_class])[0]

    print("Predicted Fracture Type:", fracture_type)
