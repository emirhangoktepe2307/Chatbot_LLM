FROM python:3.9-slim

WORKDIR /app

# Gerekli derleyici araçlarını ekle
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "chroma_setup.py"] 