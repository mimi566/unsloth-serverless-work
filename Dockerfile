# Stable RunPod base with CUDA 12.1 (we'll override PyTorch to 2.8.0+cu121 like your Colab)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# Basics
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.8.0+cu121 (exactly like your working Colab — fixes inductor.config error)
RUN pip install --upgrade pip && \
    pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Unsloth + deps (uses the new PyTorch)
COPY builder/requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 && \
    rm -rf /root/.cache/pip

# Unsloth fast kernels
ENV UNSLOTH_FORCE_CUDA=1

# Copy code
COPY src /src

CMD ["python3", "-u", "/src/handler.py"]