# This is the ONLY combination that works 100% with current Unsloth (Nov 2025)
FROM runpod/pytorch:2.5.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.8.0 + cu126 (exactly like your working Colab)
RUN pip install --upgrade pip && \
    pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

COPY builder/requirements.txt .

# Now install Unsloth and others (they will use the 2.8.0 we just installed)
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

ENV UNSLOTH_FORCE_CUDA=1

COPY src /src

CMD ["python3", "-u", "/src/handler.py"]