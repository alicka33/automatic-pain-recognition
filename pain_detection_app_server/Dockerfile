# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
    libopenblas0 \
    liblapack3 \
    liblapacke \
    libomp-dev \
    gcc \
    g++ \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch separately (CPU or GPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install MediaPipe and other ML dependencies
RUN pip install --no-cache-dir \
    mediapipe \
    numpy \
    opencv-python-headless \
    scikit-learn

# Create necessary directories for runtime
RUN mkdir -p /tmp/pain_detection_uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEDIAPIPE_DISABLE_GPU=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Expose port
EXPOSE 7860

# Run FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
