# RunPod base image with Python 3.11 + CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & setuptools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.6.0 + cu124 (adds torch.int1 for torchao compatibility)
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install Unsloth directly from git (the ONLY way that never fails on RunPod right now)
RUN pip install --no-cache-dir \
    "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"

# Install the rest of your dependencies (no Unsloth line here anymore)
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Enable Unsloth fast kernels
ENV UNSLOTH_FORCE_CUDA=1

# Copy your source code
COPY src /src

# Default command
CMD ["python3", "-u", "/src/handler.py"]