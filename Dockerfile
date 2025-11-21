# Use RunPod base image with Python 3.11 + CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

# Install git (required for pip git installs)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip & setuptools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.8.1 + cu126 (compatible with Unsloth 4-bit)
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.8.1+cu126 \
    torchvision==0.19.1+cu126 \
    torchaudio==2.8.1+cu126 \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Copy and install Unsloth + dependencies
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 && \
    rm -rf /root/.cache/pip

# Enable Unsloth fast kernels
ENV UNSLOTH_FORCE_CUDA=1

# Copy your source code
COPY src /src

# Default command
CMD ["python3", "-u", "/src/handler.py"]
