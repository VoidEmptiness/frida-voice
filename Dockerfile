FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/models/torch_hub \
    HF_HOME=/app/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        libsndfile1 \
        libsox-dev \
        sox \
        curl \
        ca-certificates \
        python3-dev \
        espeak-ng \
        espeak-ng-data \
        cmake \
        g++ \
        mecab \
        libmecab-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade "pip==24.0" "setuptools==69.5.1" "wheel"

COPY requirements.txt .

RUN pip install \
        "numpy==1.23.5" \
        "torch==2.3.1" \
        "torchaudio==2.3.1" \
    && pip install \
        "omegaconf==2.0.6" \
        "bitarray" \
        "antlr4-python3-runtime==4.8" \
        "hydra-core==1.0.7" \
        "fairseq==0.12.2" \
    && pip install -r requirements.txt \
    && pip install "piper-tts" "onnxruntime>=1.14.0,<1.20.0" "pyopenjtalk>=0.3.0"

COPY scripts/ /app/scripts/
COPY config.yaml /app/config.yaml

RUN mkdir -p /app/models /app/output /app/tmp /app/models/torch_hub

ENTRYPOINT ["python", "/app/scripts/tts.py"]
CMD ["--interactive"]
