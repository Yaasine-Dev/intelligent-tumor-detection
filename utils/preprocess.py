import os

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet import preprocess_input

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image(image_path: str, target_size: int = 300) -> np.ndarray:
    """
    Charge une image IRM et la redimensionne.

    Args:
        image_path:  chemin vers le fichier image.
        target_size: taille cible (carré), défaut 300.

    Returns:
        numpy array uint8 de shape (target_size, target_size, 3).

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError:        si le format n'est pas supporté.
        ValueError:        si l'image est corrompue ou illisible.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Format '{ext}' non supporté. Formats acceptés : {', '.join(SUPPORTED_FORMATS)}"
        )

    try:
        img = Image.open(image_path).convert("RGB").resize(
            (target_size, target_size), Image.LANCZOS
        )
    except UnidentifiedImageError:
        raise ValueError(f"Impossible de lire l'image (fichier corrompu ou format invalide) : {image_path}")

    return np.array(img, dtype=np.uint8)


def preprocess_for_model(img_array: np.ndarray) -> np.ndarray:
    """
    Prétraite un array image pour l'inférence EfficientNetB3.

    Args:
        img_array: array uint8 de shape (H, W, 3).

    Returns:
        array float32 de shape (1, H, W, 3) prêt pour model.predict().
    """
    arr = np.expand_dims(img_array.astype(np.float32), axis=0)  # (1, H, W, 3)
    return preprocess_input(arr)


def preprocess_for_gradcam(img_array: np.ndarray) -> np.ndarray:
    """
    Prétraite un array image pour Grad-CAM (même pipeline que l'inférence).

    Args:
        img_array: array uint8 de shape (H, W, 3).

    Returns:
        array float32 de shape (1, H, W, 3) prêt pour GradCAM.generate().
    """
    return preprocess_for_model(img_array)


# ── Rétrocompatibilité ─────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """Charge et prétraite une image en une seule étape (rétrocompatibilité)."""
    return preprocess_for_model(load_image(image_path))
