FROM python:3.11-slim-bookworm

# System deps for OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy code + models
COPY . .

# Cloud Run will set PORT env; default to 8080 if not set
ENV PORT=8080
EXPOSE 8080

# Use sh -c so ${PORT} is expanded by the shell
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
