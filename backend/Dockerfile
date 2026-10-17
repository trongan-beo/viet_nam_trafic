FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system dependencies for OpenCV, PaddleOCR, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install numpy and lightweight PyTorch CPU (saves ~3GB of CUDA wheels & avoids disk errors)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.26.4" && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir paddlepaddle==2.6.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ || pip install --no-cache-dir paddlepaddle==2.6.1

# Install FastAPI, Ultralytics, and PaddleOCR
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.30.1" \
    python-multipart==0.0.9 \
    opencv-python-headless==4.9.0.80 \
    "ultralytics>=8.3.0" \
    paddleocr==2.8.1 \
    pillow==10.3.0

# Copy application source code
COPY . /app

# Ensure directories for uploads, evidence, and models exist
RUN mkdir -p uploads evidence models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
