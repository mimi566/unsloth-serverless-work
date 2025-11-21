# THIS ONE WORKS 100% — DO NOT CHANGE ANYTHING
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

# System deps
RUN apt-get update && apt-get install -y git cmake build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Force PyTorch 2.5.0 + cu124 (fixes the inductor.config error)
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install Unsloth directly from git (the ONLY way that never fails on RunPod right now)
RUN pip install --no-cache-dir \
    "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"

# Install the rest of your dependencies (no Unsloth line here anymore)
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Unsloth fast kernels
ENV UNSLOTH_FORCE_CUDA=1

# Copy code
COPY src /src

CMD ["python3", "-u", "/src/handler.py"]