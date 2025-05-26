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

# Install Python dependencies (huggingface-hub is in requirements.txt, so it will be available)
# Using --no-cache-dir to potentially reduce image size slightly, though --cache-dir during multi-stage builds can be useful.
# Sticking with user's --cache-dir as it might be intentional for their build environment.
RUN pip3 install --cache-dir=/root/.cache/pip -r requirements.txt

# Copy the rest of the application's source code from the current directory to the working directory in the container
# This is moved BEFORE the checkpoint download to prevent overwriting the checkpoints.
COPY . .

# Create checkpoints directory, download models, and verify
# This step is now AFTER copying the application code.
# Includes verbose logging and verification.
RUN echo "Creating checkpoints directory..." && \
    mkdir -p /app/checkpoints && \
    echo "Downloading ByteDance/LatentSync-1.5: whisper/tiny.pt to /app/checkpoints/whisper/tiny.pt..." && \
    huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir /app/checkpoints --local-dir-use-symlinks False && \
    echo "Downloading ByteDance/LatentSync-1.5: latentsync_unet.pt to /app/checkpoints/latentsync_unet.pt..." && \
    huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir /app/checkpoints --local-dir-use-symlinks False && \
    echo "" && \
    echo "--- Contents of /app/checkpoints after download: ---" && \
    ls -lR /app/checkpoints && \
    echo "----------------------------------------------------" && \
    echo "" && \
    echo "Verifying downloaded checkpoint files..." && \
    FILE1_PATH="/app/checkpoints/whisper/tiny.pt" && \
    FILE2_PATH="/app/checkpoints/latentsync_unet.pt" && \
    if [ -f "$FILE1_PATH" ]; then \
        echo "SUCCESS: Checkpoint file $FILE1_PATH found."; \
    else \
        echo "ERROR: Checkpoint file $FILE1_PATH NOT found! Download may have failed or the path/filename within the repo is incorrect."; \
        exit 1; \
    fi && \
    if [ -f "$FILE2_PATH" ]; then \
        echo "SUCCESS: Checkpoint file $FILE2_PATH found."; \
    else \
        echo "ERROR: Checkpoint file $FILE2_PATH NOT found! Download may have failed or the path/filename within the repo is incorrect."; \
        exit 1; \
    fi && \
    echo "All specified checkpoint files verified successfully."

# Make port 7860 available to the world outside this container (Gradio default)
EXPOSE 7860

# Define environment variable
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run gradio_app.py when the container launches
CMD ["python3", "gradio_app.py"]
