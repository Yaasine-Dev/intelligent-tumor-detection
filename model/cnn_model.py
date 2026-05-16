import os
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image

CLASSES = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'tumor_model.keras')


class CNNModel:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        img = preprocess_image(image_path)
        scores = self.model.predict(img, verbose=0)[0]
        idx = int(np.argmax(scores))

        return {
            'class': CLASSES[idx],
            'confidence': round(float(scores[idx]) * 100, 2),
            'all_scores': {cls: round(float(s) * 100, 2) for cls, s in zip(CLASSES, scores)},
        }
