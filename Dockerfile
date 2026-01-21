FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Dépendances système pour l'audio et OpenCV
# libgl1 et libglib2.0-0 sont souvent nécessaires pour opencv-python-headless
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libavcodec-extra \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# On installe d'abord les dépendances pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# On copie le reste du code (y compris les fichiers .pth)
COPY . .

# Port exposé par FastAPI
EXPOSE 8000

# Commande de lancement (assure-toi que ton fichier python s'appelle bien app.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
