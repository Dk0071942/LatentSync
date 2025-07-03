# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LatentSync is an end-to-end lip-sync method based on audio-conditioned latent diffusion models. It generates lip-synced videos by directly modeling audio-visual correlations in the latent space of Stable Diffusion, without requiring intermediate motion representations.

## Key Technologies

- **PyTorch 2.7.0** with CUDA 12.1
- **Diffusers 0.32.2** for diffusion model implementation
- **Whisper** for audio encoding
- **Gradio 5.24.0** for web interface
- **RIFE** (as submodule) for frame interpolation
- **InsightFace** for face detection

## Essential Commands

### Environment Setup
```bash
# Complete setup (conda env, dependencies, checkpoints)
source setup_env.sh
```

### Running Inference
```bash
# Web Interface
python gradio_app.py

# Command Line (with demo files)
./inference.sh

# Custom inference
python -m scripts.inference \
    --unet_config_path "configs/unet/stage2_512.yaml" \
    --inference_ckpt_path "checkpoints/default_unet_v1.5.pt" \
    --video_path "path/to/video.mp4" \
    --audio_path "path/to/audio.wav" \
    --video_out_path "output.mp4"
```

### Training
```bash
# Train UNet (configure in configs/unet/*.yaml)
./train_unet.sh

# Train SyncNet (configure in configs/syncnet/*.yaml)
./train_syncnet.sh

# Fine-tuning
./fine_tuning.sh
```

### Data Processing
```bash
# Full preprocessing pipeline
./data_processing_pipeline.sh

# Generate file lists for training
python -m tools.write_fileslist
```

### Evaluation
```bash
# Evaluate sync confidence
./eval/eval_sync_conf.sh

# Evaluate SyncNet accuracy
./eval/eval_syncnet_acc.sh
```

### Docker Deployment
```bash
# Build and run Docker container
docker build -t latentsync .
docker run -p 8000:8000 --gpus all latentsync
```

## Architecture Overview

### Core Components

1. **Diffusion Pipeline** (`latentsync/pipelines/`)
   - Modified UNet3D with audio conditioning via cross-attention
   - Operates in VAE latent space for efficiency
   - Supports DeepCache for faster inference

2. **Audio Processing** (`latentsync/whisper/`)
   - Audio → Melspectrogram → Whisper Encoder → Audio Embeddings
   - Embeddings integrated into UNet cross-attention layers

3. **Model Architecture** (`latentsync/models/`)
   - `unet_3d_blocks.py`: Core UNet blocks with temporal layers
   - `syncnet_model.py`: Lip-sync discriminator
   - `attention.py`: Custom attention mechanisms
   - `resnet.py`: Residual blocks with temporal support

4. **Loss Functions** (`latentsync/trepa/`)
   - TREPA loss for temporal consistency
   - LPIPS for perceptual quality
   - SyncNet loss for lip-sync accuracy

### Training Strategy

- **Stage 1**: Initial training at 256x256 resolution
- **Stage 2**: Enhanced training with additional temporal layers
- **512px variants**: Higher resolution for production quality

### Key Directories

- `/configs/`: All configuration files (audio, scheduler, model configs)
- `/checkpoints/`: Model weights (downloaded by setup_env.sh)
- `/scripts/`: Main execution scripts for training and inference
- `/tools/`: Utility scripts for data processing
- `/preprocess/`: Video preprocessing pipeline components
- `/eval/`: Evaluation metrics and scripts
- `/ECCV2022-RIFE/`: Frame interpolation submodule

## Technical Pipeline Details

### Detailed Pipeline Workflow

1. **Audio Feature Extraction**:
   - Whisper encoder (tiny: 384-dim, small: 768-dim) extracts features at 50 FPS
   - Temporal windowing with configurable overlap (default: [2, 2])
   - Features are cached to disk for efficiency
   - Audio embeddings fed to UNet via cross-attention

2. **Video Processing**:
   - Face detection using InsightFace/MediaPipe
   - Affine transformation normalizes face position/orientation
   - Fixed 256x256 mouth region masks
   - Handles missing faces with face_detected_flags
   - Supports video looping for longer audio

3. **Diffusion Process**:
   - VAE converts images to 4-channel latents (scale: 0.18215)
   - UNet input: 13 channels [noisy_latents(4) + mask(1) + masked_image(4) + reference_image(4)]
   - DDIM scheduler with 20 steps (configurable)
   - Processes 16 frames per batch
   - Classifier-free guidance scale: 1.0-3.0

4. **UNet Architecture**:
   - 3D UNet with channels: [320, 640, 1280, 1280]
   - Cross-attention layers for audio-visual fusion
   - FlashAttention-2 via F.scaled_dot_product_attention
   - Motion modules implemented but unused in final version
   - Zero-initialized conv_in/conv_out for stability

### Current Optimizations

1. **Performance Optimizations**:
   - FlashAttention integration for efficient attention computation
   - DeepCache support (cache_interval: 3, branch_id: 0)
   - Mixed precision training (FP16 with FP32 fallback)
   - Gradient checkpointing for memory efficiency
   - Audio embedding caching

2. **Memory Optimizations**:
   - Memmap for video restoration
   - VAE tiling support for large images
   - Batch size 1 to fit consumer GPUs

### Performance Bottlenecks

1. **Computational Bottlenecks**:
   - Sequential processing of 16-frame chunks (no parallelization)
   - Face detection is CPU-bound and sequential
   - Multiple FFmpeg subprocess calls
   - Per-frame RealESRGAN upscaling

2. **Memory Bottlenecks**:
   - 13-channel concatenated tensors
   - No memory pooling/reuse
   - Fixed batch size (no dynamic batching)
   - Large intermediate tensor storage

3. **I/O Bottlenecks**:
   - Multiple disk writes during processing
   - Temporary file creation/deletion
   - No streaming inference

### Potential Quality Improvements

1. **Temporal Consistency**:
   - Enable motion modules for better temporal coherence
   - Implement temporal losses (currently unused)
   - Extend temporal receptive field beyond 16 frames
   - Add optical flow guidance

2. **Audio-Visual Sync**:
   - Implement multi-scale temporal alignment
   - Increase audio context window
   - Add learnable audio-visual attention
   - Stronger sync loss weight (currently 0.05)

3. **Visual Quality**:
   - Implement learned mask generation (adaptive to face shape)
   - Add adversarial training with discriminator
   - Multi-resolution training strategy
   - Better boundary blending for masks

4. **Model Architecture**:
   - Add skip connections between encoder/decoder
   - Implement attention mechanisms at multiple scales
   - Use continuous time embeddings
   - Add style/identity preservation modules

### Potential Speed Improvements

1. **Parallelization**:
   ```python
   # Batch multiple video chunks
   # Current: process chunks sequentially
   # Improved: process N chunks in parallel
   ```
   - Parallel face detection using multiprocessing
   - Concurrent audio feature extraction
   - Batch affine transformations on GPU

2. **Model Optimizations**:
   - Add torch.compile() for 20-30% speedup
   - Implement xFormers for additional attention speedup
   - Use CUDA graphs for static computation
   - Quantization (INT8/FP8) for inference

3. **Pipeline Optimizations**:
   - In-memory video processing (avoid disk I/O)
   - Streaming inference without full video load
   - KV-cache for cross-attention layers
   - Reuse face detection across similar frames

4. **Inference Optimizations**:
   - Reduce DDIM steps (20 → 10-15) with better scheduler
   - Implement latent consistency models
   - Use TensorRT for deployment
   - Add token merging for attention layers

### Technical Limitations

1. **Architecture Constraints**:
   - Fixed 256x256 processing resolution
   - 16-frame temporal window limitation
   - No adaptive computation based on complexity
   - Limited to frontal/near-frontal faces

2. **Quality Constraints**:
   - Upscaling artifacts from RealESRGAN
   - Fixed mask boundaries cause blending issues
   - No identity preservation guarantees
   - Sensitive to extreme head poses

3. **Performance Constraints**:
   - High VRAM requirements (20-55GB)
   - Sequential processing paradigm
   - No real-time capability
   - Limited batch processing support

## Development Guidelines

### Working with Configurations
- Model configurations are in YAML format under `/configs/`
- Audio settings: `configs/audio.yaml`
- Scheduler settings: `configs/scheduler_config.json`
- Training configs specify model architecture, loss weights, and data paths

### Adding New Features
- Model modifications go in `latentsync/models/`
- Pipeline changes in `latentsync/pipelines/`
- Data processing utilities in `latentsync/utils/`
- Training scripts should follow the pattern in `/scripts/`

### Memory Requirements
- Inference: 20-55GB VRAM depending on resolution and batch size
- Training: 40GB+ VRAM recommended
- Use `--vram_efficient` flags for consumer GPUs

### Common Issues
- Ensure CUDA 12.1 compatibility for PyTorch
- Face detection requires proper mediapipe installation
- Audio processing needs librosa and correct sample rates
- RIFE submodule must be initialized for frame interpolation

### Optimization Priorities

**High Priority** (Easy wins, high impact):
1. Add torch.compile() to UNet and VAE
2. Implement batched inference for video chunks
3. Enable xFormers/Flash-3 attention
4. Add CUDA streams for parallel operations

**Medium Priority** (Moderate effort, good impact):
1. Implement learned mask generation
2. Add temporal consistency losses
3. Use gradient accumulation for larger batches
4. Implement model quantization

**Low Priority** (High effort or experimental):
1. Switch to more efficient schedulers (DPM++, UniPC)
2. Implement progressive resolution training
3. Add model distillation
4. Experiment with ControlNet integration