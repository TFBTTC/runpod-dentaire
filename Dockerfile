# Étape 1 : image de base
FROM python:3.10-slim

# Étape 2 : installation des dépendances système
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Étape 3 : création du dossier de travail
WORKDIR /app

# Étape 4 : copie isolée de requirements.txt
COPY requirements.txt .

# Étape 5 : installation des packages Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 6 : copie du reste du projet
COPY . .

ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1


# Étape 7 : point d’entrée
CMD ["python", "handler.py"]



