# Use an official Python runtime as a parent image
# Consider using a CUDA-enabled base image if GPU support is critical, e.g., nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# This is suggested by the presence of onnxruntime-gpu and torch with cu121 in requirements.txt
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install git, ffmpeg, and libgl1 (for OpenCV), and build-essential (for g++ and other build tools), python3 and python3-pip
# Added python3-dev for potential C extensions in Python packages
RUN apt-get update && \
    apt-get install -y git ffmpeg libgl1 build-essential python3 python3-pip python3-dev --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Copy the rest of the application code into the container
COPY . .

# Initialize git submodules
RUN git submodule init && git submodule update

# Install dependencies, including huggingface_hub for model downloads
RUN pip install --no-cache-dir -r requirements.txt && pip install huggingface_hub

# Create checkpoints directory, download models, and verify
RUN echo "Creating checkpoints directory..." && \
    mkdir -p /app/checkpoints && \
    echo "Downloading checkpoints..." && \
    huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir /app/checkpoints --local-dir-use-symlinks False && \
    huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir /app/checkpoints --local-dir-use-symlinks False && \
    huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir /app/checkpoints --local-dir-use-symlinks False && \
    mv /app/checkpoints/latentsync_unet.pt /app/checkpoints/default_unet_v1.5.pt && \
    huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir /app/checkpoints --local-dir-use-symlinks False && \
    mv /app/checkpoints/latentsync_unet.pt /app/checkpoints/default_unet_v1.6.pt && \
    echo "" && \
    echo "--- Contents of /app/checkpoints after download: ---" && \
    ls -lR /app/checkpoints && \
    echo "----------------------------------------------------" && \
    echo "" && \
    echo "Verifying downloaded checkpoint files..." && \
    FILE1_PATH="/app/checkpoints/whisper/tiny.pt" && \
    FILE2_PATH="/app/checkpoints/stable_syncnet.pt" && \
    FILE3_PATH="/app/checkpoints/default_unet_v1.5.pt" && \
    FILE4_PATH="/app/checkpoints/default_unet_v1.6.pt" && \
    if [ -f "$FILE1_PATH" ] && [ -f "$FILE2_PATH" ] && [ -f "$FILE3_PATH" ] && [ -f "$FILE4_PATH" ]; then \
        echo "SUCCESS: All checkpoint files found."; \
    else \
        echo "ERROR: One or more checkpoint files are missing!"; \
        [ -f "$FILE1_PATH" ] || echo "Missing: $FILE1_PATH"; \
        [ -f "$FILE2_PATH" ] || echo "Missing: $FILE2_PATH"; \
        [ -f "$FILE3_PATH" ] || echo "Missing: $FILE3_PATH"; \
        [ -f "$FILE4_PATH" ] || echo "Missing: $FILE4_PATH"; \
        exit 1; \
    fi && \
    echo "All specified checkpoint files verified successfully."

# Make port 7860 available to the world outside this container (Gradio default)
EXPOSE 7860

# Define environment variable
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run gradio_app.py when the container launches
CMD ["python3", "gradio_app.py"]
