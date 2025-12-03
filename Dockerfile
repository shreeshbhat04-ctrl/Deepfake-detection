# 1. Use stable python image
FROM python:3.11-slim-bookworm

# 2. Set up a non-root user (Good for security on all clouds)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# 3. Set working directory
WORKDIR $HOME/app

# 4. Install system dependencies (as root)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*
USER user

# 5. Upgrade pip
RUN pip install --upgrade pip

# 6. Copy requirements
COPY --chown=user requirements.txt .

# 7. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy the rest of the application code
COPY --chown=user . .

# 9. UNIVERSAL PORT CONFIGURATION
ENV PORT=8080
EXPOSE 8080

# 10. Start the app using the variable
# We use the shell form so $PORT is definitely read
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
