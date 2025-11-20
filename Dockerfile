FROM runpod/pytorch:2.4.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

ENV UNSLOTH_FORCE_CUDA=1

COPY src /src

CMD ["python3", "-u", "/src/handler.py"]