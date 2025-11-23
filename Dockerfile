FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Basic system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install faster-whisper from PyPI (official package from SYSTRAN)
RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir faster-whisper

# Copy our CLI script
COPY transcribe.py /app/transcribe.py
RUN chmod +x /app/transcribe.py

ENTRYPOINT ["python3", "/app/transcribe.py"]
