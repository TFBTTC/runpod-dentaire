# handler.py
import requests
import cv2
import numpy as np
from main_logic import process_cv2_image  # renomme `main.py` en `main_logic.py` si besoin

def handler(event):
    image_url = event.get("image_url")
    if not image_url:
        return {"error": "Il manque le champ 'image_url'"}

    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Téléchargement échoué: {str(e)}"}

    image_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Échec de la lecture de l'image"}

    result = process_cv2_image(img)

    return {
        "message": "Analyse terminée ✅",
        "summary": result
    }
