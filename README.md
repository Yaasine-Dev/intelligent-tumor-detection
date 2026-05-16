# 🧠 Intelligent Brain Tumor Detection System

Application de bureau utilisant le Deep Learning (TensorFlow/Keras) pour la détection de tumeurs cérébrales et un LLM (Groq API) pour l'analyse intelligente et la génération de rapports médicaux.

## ✨ Fonctionnalités

- 🔍 **Détection automatique** de 4 types de tumeurs cérébrales (Gliome, Méningiome, Pituitaire, Pas de tumeur)
- 🎯 **EfficientNetB3** : Modèle CNN pré-entraîné avec ~11.7M paramètres
- 🔥 **Grad-CAM** : Visualisation des zones d'attention du modèle
- 🤖 **LLM Groq** : Génération de rapports médicaux détaillés en français
- 🖥️ **Interface PyQt5** : Interface graphique moderne avec drag & drop
- 📄 **Export PDF** : Rapports professionnels avec images et analyses

## 🛠️ Stack Technique

- **Python 3.12**
- **TensorFlow 2.21.0** / Keras 3.14.1
- **PyQt5 5.15.11**
- **Groq API** (Llama-3.3-70b-versatile)
- **ReportLab 4.5.0**
- **OpenCV 4.9.0.80**

## 📦 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer l'API Groq
```bash
cp .env.example .env
```
Éditez `.env` et ajoutez votre clé API Groq :
```
GROQ_API_KEY=votre_clé_api_ici
```
> Obtenez votre clé sur : https://console.groq.com

### 5. Vérifier le modèle
Assurez-vous que `tumor_model.keras` est présent dans le dossier `model/`

## 🚀 Utilisation

### Lancer l'application
```bash
python main.py
```

### Tester le modèle
```bash
python test_model.py
```

### Workflow
1. **Charger une IRM** : Glissez-déposez ou cliquez sur "Charger IRM"
2. **Analyser** : Cliquez sur "Analyser" pour lancer la détection
3. **Visualiser** : Consultez l'image originale, le Grad-CAM et les résultats
4. **Exporter** : Générez un rapport PDF professionnel

## 📊 Architecture du Projet

```
.
├── model/
│   ├── cnn_model.py          # Classe CNN pour prédictions
│   ├── gradcam.py            # Génération de heatmaps Grad-CAM
│   └── tumor_model.keras     # Modèle EfficientNetB3 pré-entraîné
├── llm/
│   ├── llm_client.py         # Client Groq API
│   └── prompt_builder.py     # Construction des prompts
├── ui/
│   ├── main_window.py        # Interface PyQt5 principale
│   └── styles.qss            # Styles CSS pour l'interface
├── utils/
│   ├── preprocess.py         # Prétraitement d'images
│   └── pdf_export.py         # Export PDF avec ReportLab
├── main.py                   # Point d'entrée de l'application
├── test_model.py             # Script de test du modèle
└── requirements.txt          # Dépendances Python
```

## 🧪 Tests

### Test du modèle CNN
```bash
python test_model.py
```
Affiche : architecture, input/output shape, nombre de paramètres, test de prédiction

### Test de l'API Groq
```bash
python test_groq.py
```

## 📝 Classes de Tumeurs

1. **Gliome** : Tumeur maligne du tissu glial
2. **Méningiome** : Tumeur des méninges (généralement bénigne)
3. **Pituitaire** : Tumeur de l'hypophyse
4. **Pas de tumeur** : IRM normale

## 🎯 Résultats du Modèle

- **Architecture** : EfficientNetB3 (392 couches)
- **Input Shape** : (300, 300, 3)
- **Paramètres** : 11,709,021 (6.2M entraînables)
- **Classes** : 4

## ⚠️ Avertissement Médical

Ce système est un **outil d'aide à la décision** uniquement. Il ne remplace pas l'avis d'un médecin qualifié. Toute décision thérapeutique doit être prise par un professionnel de santé habilité.

## 📄 Licence

MIT License

## 👥 Contributeurs

- **Abdessattar** : Entraînement du modèle CNN
- **Najoua** : Interface PyQt5, intégration LLM, export PDF

## 🔗 Liens Utiles

- [TensorFlow](https://www.tensorflow.org/)
- [Groq API](https://console.groq.com/)
- [PyQt5 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
