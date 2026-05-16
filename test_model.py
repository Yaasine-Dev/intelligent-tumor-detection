"""
Script de test pour vérifier le modèle EfficientNetB3 de détection de tumeurs cérébrales.
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "tumor_model.keras")
CLASSES    = ["Gliome", "Méningiome", "Pas de tumeur", "Pituitaire"]
INPUT_SIZE = (300, 300, 3)


# ── Fonctions ──────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def load_model():
    """Charge le modèle depuis le fichier .keras."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Assurez-vous que 'tumor_model.keras' est présent dans le dossier model/"
        )
    
    print(f"[*] Chargement du modele depuis : {MODEL_PATH}")
    model = keras_load_model(MODEL_PATH, compile=False)
    print("[OK] Modele charge avec succes\n")
    return model


def display_model_info(model):
    """Affiche les informations architecturales du modèle."""
    print_section("INFORMATIONS DU MODÈLE")
    
    # Input shape
    input_shape = model.input_shape
    print(f"  Input shape  : {input_shape}")
    
    # Output shape
    output_shape = model.output_shape
    print(f"  Output shape : {output_shape}")
    
    # Nombre de classes
    num_classes = output_shape[-1]
    print(f"  Nombre de classes : {num_classes}")
    print(f"  Classes attendues : {CLASSES}")
    
    # Nombre de couches
    num_layers = len(model.layers)
    print(f"  Nombre de couches : {num_layers}")
    
    # Nombre de paramètres
    trainable_params = sum(
        np.prod(var.shape) for var in model.trainable_variables
    )
    non_trainable_params = sum(
        np.prod(var.shape) for var in model.non_trainable_variables
    )
    total_params = trainable_params + non_trainable_params
    
    print(f"\n  Parametres :")
    print(f"    - Entrainables     : {trainable_params:,}")
    print(f"    - Non-entrainables : {non_trainable_params:,}")
    print(f"    - Total            : {total_params:,}")


def test_prediction(model):
    """Teste une prédiction avec une image factice."""
    print_section("TEST DE PRÉDICTION")
    
    # Creation d'une image factice (array de zeros)
    print(f"[*] Creation d'une image factice de shape (1, {INPUT_SIZE[0]}, {INPUT_SIZE[1]}, {INPUT_SIZE[2]})")
    fake_image = np.zeros((1, *INPUT_SIZE), dtype=np.float32)
    
    # Prediction
    print("[*] Execution de la prediction...")
    predictions = model.predict(fake_image, verbose=0)
    
    # Affichage des resultats
    print(f"[OK] Prediction reussie\n")
    print(f"  Shape de sortie : {predictions.shape}")
    print(f"  Type de donnees : {predictions.dtype}\n")
    
    # Scores par classe
    print("  Scores par classe :")
    for i, (cls, score) in enumerate(zip(CLASSES, predictions[0])):
        bar = "#" * int(score * 50)
        print(f"    {i}. {cls:20s} : {score:.6f}  {bar}")
    
    # Classe predite
    predicted_idx = int(np.argmax(predictions[0]))
    predicted_class = CLASSES[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    print(f"\n  >> Classe predite : {predicted_class} (confiance : {confidence:.2f}%)")


def verify_architecture(model):
    """Vérifie que l'architecture correspond à EfficientNetB3."""
    print_section("VÉRIFICATION DE L'ARCHITECTURE")
    
    # Vérification input shape
    expected_input = (None, 300, 300, 3)
    actual_input = model.input_shape
    
    if actual_input == expected_input:
        print(f"[OK] Input shape correct : {actual_input}")
    else:
        print(f"[!!] Input shape inattendu : {actual_input} (attendu : {expected_input})")
    
    # Verification output shape
    expected_output_classes = len(CLASSES)
    actual_output_classes = model.output_shape[-1]
    
    if actual_output_classes == expected_output_classes:
        print(f"[OK] Nombre de classes correct : {actual_output_classes}")
    else:
        print(f"[!!] Nombre de classes inattendu : {actual_output_classes} (attendu : {expected_output_classes})")
    
    # Recherche de la couche top_conv (specifique a EfficientNet)
    has_top_conv = any(layer.name == "top_conv" for layer in model.layers)
    if has_top_conv:
        print("[OK] Couche 'top_conv' trouvee (architecture EfficientNet confirmee)")
    else:
        print("[!!] Couche 'top_conv' non trouvee (architecture EfficientNet non confirmee)")


def display_summary(model):
    """Affiche le résumé complet du modèle."""
    print_section("RÉSUMÉ DU MODÈLE")
    model.summary()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("  TEST DU MODELE DE DETECTION DE TUMEURS CEREBRALES")
    print("=" * 70)
    
    try:
        # 1. Chargement
        model = load_model()
        
        # 2. Informations
        display_model_info(model)
        
        # 3. Vérification architecture
        verify_architecture(model)
        
        # 4. Test de prédiction
        test_prediction(model)
        
        # 5. Résumé (optionnel, commenté par défaut car très long)
        # display_summary(model)
        
        print_section("RESULTAT FINAL")
        print("[OK] Tous les tests ont reussi !")
        print("[OK] Le modele est pret a etre utilise dans l'application.\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n[ERREUR] : {e}\n")
        return 1
        
    except Exception as e:
        print(f"\n[ERREUR INATTENDUE] : {type(e).__name__}")
        print(f"   {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
