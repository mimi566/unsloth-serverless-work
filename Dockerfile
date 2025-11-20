# ─────────────────────────────────────────────────────────────
# Official RunPod PyTorch base (CUDA 12.4 + torch 2.6.0 pre-installed)
# This is the FASTEST & MOST RELIABLE base as of Nov 2025
# ─────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.4.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /

# Update & install basics
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (maximizes Docker layer caching)
COPY builder/requirements.txt .

# Install everything in one layer (fastest cold starts)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 && \
    rm -rf /root/.cache/pip

# ─────────────────────────────────────────────────────────────
# Critical Unsloth environment flags (forces fastest kernels)
# ─────────────────────────────────────────────────────────────
ENV UNSLOTH_FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

# Copy source code
COPY src /src

# Download model at build time if MODEL_NAME is passed (optional but recommended for faster first cold start)
ARG MODEL_NAME="Sourabh66/Llama-2-17B-Fine-Tune-Blog"
ARG MAX_SEQ_LENGTH=32768
ARG LOAD_IN_4BIT=true

ENV MODEL_NAME=$MODEL_NAME \
    MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH \
    LOAD_IN_4BIT=$LOAD_IN_4BIT \
    BASE_PATH="/runpod-volume"

# Optional: Pre-download model during build (saves ~20s on first cold start)
RUN python3 -c "\
from unsloth import FastLanguageModel; \
print('Pre-downloading model...'); \
model, tokenizer = FastLanguageModel.from_pretrained( \
    model_name='$MODEL_NAME', \
    max_seq_length=$MAX_SEQ_LENGTH, \
    load_in_4bit=$LOAD_IN_4BIT, \
    dtype=None \
); \
print('Model cached!') \
" || echo "Model will download on first run (if not cached)"

# Start the handler
CMD ["python3", "-u", "/src/handler.py"]