FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
 PYTHONDONTWRITEBYTECODE=1 \
 PIP_NO_CACHE_DIR=off \
 PIP_DISABLE_PIP_VERSION_CHECK=on \
 PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# "app:app" = fichier app.py / objet FastAPI nommé app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
