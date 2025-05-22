# Use an official Python runtime as a parent image
# Consider using a CUDA-enabled base image if GPU support is critical, e.g., nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# This is suggested by the presence of onnxruntime-gpu and torch with cu121 in requirements.txt
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install git, ffmpeg, and libgl1 (for OpenCV), and build-essential (for g++ and other build tools)
RUN apt-get update && \
    apt-get install -y git ffmpeg libgl1 build-essential --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Python dependencies (huggingface-hub is in requirements.txt, so it will be available)
RUN pip install --cache-dir=/root/.cache/pip -r requirements.txt

# Copy the rest of the application's source code from the current directory to the working directory in the container
COPY . .

# Make port 7860 available to the world outside this container (Gradio default)
EXPOSE 7860

# Define environment variable
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run gradio_app.py when the container launches
CMD ["python", "gradio_app.py"] 