FROM runpod/pytorch:2.4.0-py3.10-cuda12.6.1-devel

WORKDIR /

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu126

COPY builder/requirements.txt .
RUN pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 && \
    rm -rf /root/.cache/pip

ENV UNSLOTH_FORCE_CUDA=1

COPY src /src

CMD ["python3", "-u", "/src/handler.py"]
