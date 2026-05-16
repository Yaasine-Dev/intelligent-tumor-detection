# convert_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model

print(f"TensorFlow version : {tf.__version__}")

# Charger le modèle (peut échouer à cause de la compatibilité)
try:
    model = load_model('model/tumor_model.keras')
    print("✅ Modèle .keras chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur chargement .keras : {e}")
    print("\n🔄 Tentative avec compile=False...")
    try:
        model = load_model('model/tumor_model.keras', compile=False)
        print("✅ Modèle chargé avec compile=False !")
    except Exception as e2:
        print(f"❌ Échec total : {e2}")
        exit(1)

# Sauvegarder en .h5
model.save('model/tumor_model.h5')
print("✅ Modèle converti en tumor_model.h5")

# Vérifier
from tensorflow.keras.models import load_model as lm
test = lm('model/tumor_model.h5')
print(f"✅ Vérification OK ! Input shape : {test.input_shape}")