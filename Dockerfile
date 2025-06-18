FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG GITHUB_TOKEN

RUN apt-get update && apt-get install -y git ffmpeg libgl1 build-essential python3 python3-pip python3-dev --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the public main repo
RUN git clone https://github.com/DK0071942/LatentSync.git /app

# Use token to pull private submodules
WORKDIR /app
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/" && \
    git submodule update --init --recursive

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && pip install huggingface_hub

# Download checkpoints only if they don't already exist
RUN mkdir -p /app/checkpoints && \
    ([ -f "/app/checkpoints/whisper/tiny.pt" ] || \
        huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir /app/checkpoints --local-dir-use-symlinks False) && \
    ([ -f "/app/checkpoints/stable_syncnet.pt" ] || \
        huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir /app/checkpoints --local-dir-use-symlinks False) && \
    ([ -f "/app/checkpoints/default_unet_v1.5.pt" ] || ( \
        huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir /app/checkpoints --local-dir-use-symlinks False && \
        mv /app/checkpoints/latentsync_unet.pt /app/checkpoints/default_unet_v1.5.pt)) && \
    ([ -f "/app/checkpoints/default_unet_v1.6.pt" ] || ( \
        huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir /app/checkpoints --local-dir-use-symlinks False && \
        mv /app/checkpoints/latentsync_unet.pt /app/checkpoints/default_unet_v1.6.pt)) && \
    echo "Checkpoint verification..." && \
    [ -f "/app/checkpoints/whisper/tiny.pt" ] && \
    [ -f "/app/checkpoints/stable_syncnet.pt" ] && \
    [ -f "/app/checkpoints/default_unet_v1.5.pt" ] && \
    [ -f "/app/checkpoints/default_unet_v1.6.pt" ]

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python3", "gradio_app.py"]
