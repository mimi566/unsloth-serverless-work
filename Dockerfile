# Stable RunPod base with CUDA 12.4 + py3.11 (exists 100% — confirmed on Docker Hub Nov 21, 2025)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

# Basics
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade to PyTorch 2.5.0+cu124 (Unsloth-compatible, fixes inductor.config error)
RUN pip install --upgrade pip && \
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --extra-index-url https://download.pytorch.org/whl/cu124

# Copy requirements and install Unsloth + deps (uses the new PyTorch)
COPY builder/requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 && \
    rm -rf /root/.cache/pip

# Unsloth fast kernels
ENV UNSLOTH_FORCE_CUDA=1

# Copy code
COPY src /src

CMD ["python3", "-u", "/src/handler.py"]