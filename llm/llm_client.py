import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CLASS_CONTEXT = {
    "Gliome": "tumeur maligne issue du tissu glial, souvent agressive, nécessitant une prise en charge urgente",
    "Méningiome": "tumeur des méninges, généralement bénigne mais pouvant causer des compressions cérébrales",
    "Pituitaire": "tumeur de l'hypophyse pouvant affecter les fonctions hormonales",
    "Pas de tumeur": "aucune anomalie tumorale détectée sur cette IRM cérébrale",
}

class GroqClient:
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client  = Groq(api_key=self.api_key)

    def generate_report(self, prediction: dict) -> str:
        """
        Génère une explication médicale complète basée sur la prédiction CNN.
        prediction = {
            'class': 'Gliome',
            'confidence': 98.5,
            'all_scores': {'Gliome': 98.5, 'Méningiome': 1.0, ...}
        }
        """
        predicted_class = prediction['class']
        confidence      = prediction['confidence']
        all_scores      = prediction['all_scores']
        context         = CLASS_CONTEXT.get(predicted_class, "")

        scores_text = "\n".join(
            f"  - {cls}: {score:.1f}%"
            for cls, score in all_scores.items()
        )

        prompt = f"""Tu es un assistant médical expert en neurologie et imagerie médicale.

Un système d'IA a analysé une IRM cérébrale et produit ce résultat :

RÉSULTAT :
- Classe détectée : {predicted_class}
- Confiance       : {confidence:.1f}%
- Contexte        : {context}

Scores par classe :
{scores_text}

Fournis une analyse médicale structurée en français avec ces 5 sections :

1. **Description clinique** : type de tumeur, caractéristiques, localisation typique.
2. **Niveau de gravité** : faible / modéré / élevé / critique — justifié.
3. **Conduite à tenir** : prochaines étapes (surveillance, biopsie, chirurgie...).
4. **Examens complémentaires** : IRM gadolinium, TEP scan, biopsie, bilan hormonal...
5. **Questions clés pour le patient** : symptômes et antécédents à investiguer.

⚠️ Rappelle que ce système est une aide à la décision — pas un diagnostic officiel.
"""

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant médical expert en neurologie. "
                        "Tes réponses sont précises, structurées et en français. "
                        "Tu rappelles toujours que tes analyses sont des aides à la décision."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        return response.choices[0].message.content