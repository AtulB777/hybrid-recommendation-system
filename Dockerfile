FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY training ./training
COPY data/__init__.py data/generate_synthetic.py ./data/
RUN mkdir -p /app/artifacts /app/data/raw /app/docs

EXPOSE 8000
# Default = ONLINE serving. The OFFLINE job overrides the command (see docker-compose.yml).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
