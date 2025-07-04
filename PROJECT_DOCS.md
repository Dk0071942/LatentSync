# LatentSync Project Documentation

## 1. ARCHITECTURE OVERVIEW

### System Architecture
- **Purpose**: Real-time lip-sync video generation using audio-conditioned latent diffusion
- **Core Approach**: Diffusion model in VAE latent space with direct audio conditioning via Whisper embeddings
- **Key Technologies**: PyTorch, Diffusers, Whisper, VAE, DDIM, InsightFace, MediaPipe

### Data Flow
```
[Audio + Video] → [Preprocessing] → [Latent Diffusion] → [VAE Decode] → [Post-processing] → [Synced Video]
                        ↓                    ↓
                  [Whisper Encoder]    [UNet + Audio Cross-Attention]
```

### Major Components
- **UNet3D**: Core diffusion model with audio conditioning - `latentsync/models/unet.py`
- **SyncNet**: Lip-sync quality discriminator - `latentsync/models/syncnet.py`
- **Audio Processing**: Whisper-based feature extraction - `latentsync/whisper/`
- **Video Pipeline**: End-to-end inference system - `latentsync/pipelines/latentsync_pipeline.py`
- **Training System**: Multi-stage training with various losses - `scripts/train_unet.py`

## 2. COMPONENT MAP

### Component Dependencies
```
Audio → Whisper Encoder → Audio Embeddings → UNet Cross-Attention
Video → Face Detection → VAE Encode → Latent Space → UNet Denoising → VAE Decode → Output
                ↓                           ↓
            Mask Generation          SyncNet Evaluation
```

### Entry Points
- **Gradio Interface**: `gradio_app.py:224` - Web UI for easy access
- **CLI Inference**: `scripts/inference.py:380` - Command-line interface
- **Training**: `scripts/train_unet.py:664` - UNet training entry
- **API**: `gradio_app.py` - Gradio comes with API support

### Configuration System
- **Audio Config**: `configs/audio.yaml` - Sample rates, mel specs, window sizes
- **UNet Configs**: `configs/unet/*.yaml` - Model architecture and training settings
- **SyncNet Configs**: `configs/syncnet/*.yaml` - Discriminator configurations
- **Scheduler**: `configs/scheduler_config.json` - DDIM inference settings

## 3. KEY PATTERNS

### Code Patterns
- **Config Loading**: YAML-based with OmegaConf - Example: `scripts/train_unet.py:586`
- **Model Checkpointing**: State dict with optimizer states - Example: `scripts/train_unet.py:478`
- **Batch Processing**: 16-frame chunks with overlap - Example: `latentsync/pipelines/latentsync_pipeline.py:446`
- **Loss Combination**: Multi-objective with weights - Example: `scripts/train_unet.py:315`

### Common Abstractions
- **AudioEncoder**: Whisper wrapper for consistent audio features
- **FaceHelper**: Unified interface for face detection/tracking
- **VideoReader/Writer**: Abstraction over video I/O operations

### Anti-patterns to Avoid
- Don't process video all at once - use chunking for memory efficiency
- Don't use pixel-space diffusion - stay in latent space
- Don't ignore temporal consistency - use overlapping chunks
- Don't hardcode paths - use config files

## 4. NAVIGATION GUIDE

### Quick Find
- **Diffusion Implementation**: `latentsync/models/unet_3d_avaudioldm.py:140` - UNet3DConditionModel
- **Audio Conditioning**: `latentsync/models/unet_3d_avaudioldm.py:841` - Cross-attention mechanism
- **Inference Pipeline**: `latentsync/pipelines/latentsync_pipeline.py:317` - Main generation loop
- **Face Detection**: `latentsync/utils/face_helper.py:47` - Multi-backend face detection
- **Loss Functions**: `scripts/train_unet.py:315` - Combined loss calculation
- **Config Loading**: `scripts/train_unet.py:586` - Configuration setup

### Directory Purposes
- `/latentsync`: Core library implementation
  - `/data`: Dataset loaders for training
  - `/models`: Neural network architectures
  - `/pipelines`: Inference pipelines
  - `/utils`: Helper functions
  - `/whisper`: Audio feature extraction
- `/configs`: All configuration files
- `/scripts`: Training and inference scripts
- `/preprocess`: Data preparation tools
- `/eval`: Evaluation metrics
- `/tools`: Utility scripts

## 5. TASK PLAYBOOK

### Common Tasks

#### Adding a new loss function
1. Check existing losses in `scripts/train_unet.py:315`
2. Add loss computation in training loop
3. Update loss weights in config files
4. Add loss to tensorboard logging

#### Modifying audio processing
1. Start at `latentsync/whisper/audio2feature.py`
2. Check audio config in `configs/audio.yaml`
3. Update feature dimensions if needed
4. Ensure compatibility with UNet cross-attention

#### Changing video resolution
1. Update VAE in `latentsync/pipelines/latentsync_pipeline.py:98`
2. Modify face detection resolution in `latentsync/utils/face_helper.py`
3. Adjust UNet architecture if needed
4. Update configs for new resolution

#### Debugging inference issues
1. Start at `scripts/inference.py:380` entry point
2. Check logs in `latentsync/pipelines/latentsync_pipeline.py`
3. Common issues:
   - OOM: Reduce batch size or resolution
   - Face detection failure: Check input video quality
   - Audio sync issues: Verify audio sample rate

#### Training from scratch
1. Prepare dataset using `preprocess/` tools
2. Create file lists for train/val splits
3. Copy and modify config from `configs/unet/`
4. Start with stage1 training, then stage2
5. Monitor with tensorboard

#### Fine-tuning on custom data
1. Use `fine_tuning.sh` as template
2. Prepare small dataset (100-1000 samples)
3. Use lower learning rate (1e-6)
4. Train for fewer steps (5000-10000)

### Performance Optimization
- **Memory**: Use gradient checkpointing, reduce batch size
- **Speed**: Enable xformers, use DDIM with fewer steps
- **Quality**: Increase guidance scale, use more denoising steps

### Model Deployment
- **Docker**: Use provided Dockerfile
- **API**: Deploy with Cog using predict.py
- **Web**: Use Gradio interface
- **Batch**: Use CLI script for multiple videos