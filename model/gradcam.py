import cv2
import numpy as np
import tensorflow as tf

CONV_LAYER = "top_conv"  # dernière couche conv d'EfficientNetB3


class GradCAM:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.grad_model = self._build_grad_model()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _build_grad_model(self) -> tf.keras.Model:
        """Construit un modèle intermédiaire : entrée → (activations conv, logits)."""
        conv_layer = self._find_conv_layer()
        return tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output],
        )

    def _find_conv_layer(self) -> tf.keras.layers.Layer:
        """Retourne 'top_conv' si présente, sinon la dernière Conv2D du modèle."""
        for layer in self.model.layers:
            if layer.name == CONV_LAYER:
                return layer
        # fallback : dernière couche Conv2D
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        raise ValueError("Aucune couche convolutive trouvée dans le modèle.")

    # ── Core ───────────────────────────────────────────────────────────────────

    def generate(self, img_array: np.ndarray, class_index: int | None = None) -> np.ndarray:
        """
        Calcule la heatmap Grad-CAM normalisée.

        Args:
            img_array:   tableau prétraité de shape (1, H, W, 3).
            class_index: indice de la classe cible ; si None, utilise la classe prédite.

        Returns:
            heatmap float32 normalisée [0, 1] de shape (H_conv, W_conv).
        """
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            tape.watch(conv_outputs)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            score = predictions[:, class_index]

        grads = tape.gradient(score, conv_outputs)          # (1, h, w, c)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))     # (c,)
        cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)  # (h, w)

        cam = tf.nn.relu(cam).numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)

    # ── Overlay ────────────────────────────────────────────────────────────────

    def overlay_heatmap(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Superpose la heatmap colorisée sur l'image originale.

        Args:
            original_img: image uint8 RGB ou BGR de shape (H, W, 3).
            heatmap:      heatmap normalisée [0, 1] issue de generate().
            alpha:        opacité de la heatmap (0 = invisible, 1 = opaque).

        Returns:
            (heatmap_colored_bgr, superposition_bgr) — tous deux uint8 BGR.
        """
        h, w = original_img.shape[:2]

        # Redimensionne la heatmap à la taille de l'image originale
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

        # Convertit l'image originale en BGR si elle est en RGB
        if original_img.dtype != np.uint8:
            original_img = np.uint8(
                np.clip(original_img, 0, 255)
                if original_img.max() > 1
                else original_img * 255
            )
        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        superposition = cv2.addWeighted(heatmap_colored, alpha, img_bgr, 1 - alpha, 0)
        return heatmap_colored, superposition

    # ── Convenience ────────────────────────────────────────────────────────────

    def compute_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Rétrocompatibilité : retourne la heatmap pour la classe prédite."""
        return self.generate(image)
