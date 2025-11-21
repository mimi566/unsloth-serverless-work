# Base image with CUDA 12.4 (we'll override PyTorch to 2.8.0+cu126 like your Colab)
FROM runpod/base:ubuntu22.04-cuda12.4.1

WORKDIR /

# Install Python 3.11 + pip (base has no Python pre-installed)
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev git && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && rm get-pip.py && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.8.0+cu126 (exactly like your working Colab)
RUN pip install --upgrade pip && \
    pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Copy requirements and install Unsloth + deps (uses the new PyTorch)
COPY builder/requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 && \
    rm -rf /root/.cache/pip

# Unsloth fast kernels
ENV UNSLOTH_FORCE_CUDA=1

# Copy code
COPY src /src

CMD ["python3", "-u", "/src/handler.py"]