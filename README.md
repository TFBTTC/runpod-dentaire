# Worker RunPod pour analyse dentaire

Ce projet fournit un worker compatible avec la plateforme RunPod. Il applique les modèles de détection `dents`, `bridges` et `implants` sur l'image reçue, attribue les numéros FDI puis retourne un résumé.

## Utilisation locale

```bash
pip install -r requirements.txt
python main.py
```

`main.py` lance le handler avec une image de test.

## Déploiement RunPod

Le conteneur exécute `handler.py` qui démarre le worker via `runpod.serverless.start`. Après construction vous pouvez pousser l'image sur RunPod :

```bash
docker build -t dentaire-worker .
```
