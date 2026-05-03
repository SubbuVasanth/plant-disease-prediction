# ── Stage 1: Build ────────────────────────────────────────────────────────
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by Pillow / torch
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the uploads directory exists
RUN mkdir -p static/uploads

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
