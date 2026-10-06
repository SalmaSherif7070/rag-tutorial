# Image for the RAG API service.
FROM python:3.11-slim

# Keep Python output unbuffered so logs appear immediately in `docker compose logs`.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application package.
COPY rag ./rag

EXPOSE 8000

# Start the API. 0.0.0.0 so it's reachable from outside the container.
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
