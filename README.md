# Worker RunPod pour analyse dentaire

Ce projet fournit un worker compatible avec la plateforme RunPod. Il applique les modèles de détection `dents`, `bridges` et `implants` sur l'image reçue, attribue les numéros FDI puis passe chaque dent à travers trois modèles de classification (`classes_dent`, `endo`, `restauration`). Les dents détectées sont recadrées en `224x224` avec des bandes noires pour conserver le ratio avant d'être classées. Le handler retourne ensuite un résumé complet.

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
