# Python 3.12 slim image
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Point pip to the cu124 wheels for PyTorch
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

# System deps often needed by scientific stacks and matplotlib backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libopenblas-dev liblapack-dev \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Quick CUDA sanity check at startup (optional)
CMD ["python", "-c", "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| Available:', torch.cuda.is_available())"]
