#!/bin/bash

# Check if the environment 'latentsync' already exists
if conda env list | grep -q '^latentsync\s'; then
    echo "Environment 'latentsync' already exists. Skipping creation."
else
    conda create -y -n latentsync python=3.10.13
fi

# Activate the environment
conda activate latentsync

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Python dependencies
pip install -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints
mv checkpoints/latentsync_unet.pt checkpoints/default_unet_v1.5.pt
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints
mv checkpoints/latentsync_unet.pt checkpoints/default_unet_v1.6.pt

git submodule init
git submodule update