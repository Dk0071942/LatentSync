FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

ARG GITHUB_TOKEN

# Install FFmpeg 7.1 to match local development environment
RUN apt-get update && apt-get install -y git wget xz-utils libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 build-essential python3 python3-pip python3-dev --no-install-recommends \
    && wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -O /tmp/ffmpeg.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg* \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the application code from the build context
COPY . .

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

EXPOSE 8000
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Authentication Environment Variables
# Set to empty strings to disable authentication (default for local development)
# Override these in deployment to enable authentication
ENV AUTH_USERNAME=""
ENV AUTH_PASSWORD=""

CMD ["python3", "gradio_app.py"]
