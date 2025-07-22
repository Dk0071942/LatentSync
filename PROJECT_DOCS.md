# LatentSync Project Documentation

## 1. ARCHITECTURE OVERVIEW

### System Architecture
- **Purpose**: Real-time lip-sync video generation using audio-conditioned latent diffusion models
- **Core Approach**: End-to-end latent space diffusion with direct audio conditioning via Whisper embeddings
- **Key Technologies**: PyTorch, Diffusers, Stable Diffusion, Whisper, VAE, DDIM, InsightFace, MediaPipe
- **Innovation**: First method to use latent diffusion for lip-sync without intermediate motion representation

### System Architecture Diagram

```mermaid
graph TB
    subgraph "Input Processing"
        VI[Input Video] --> FD[Face Detection<br/>InsightFace/MediaPipe]
        AI[Input Audio] --> WE[Whisper Encoder<br/>384/768-dim features]
        FD --> AT[Affine Transform<br/>Face Normalization]
        AT --> MG[Mask Generation<br/>256x256 mouth region]
    end
    
    subgraph "Latent Diffusion Pipeline"
        MG --> VAE_E[VAE Encoder<br/>4-channel latents]
        VAE_E --> CONCAT[Channel Concatenation<br/>13 channels total]
        WE --> CA[Cross-Attention<br/>Audio-Visual Fusion]
        CONCAT --> UNET[3D UNet<br/>Audio-Conditioned]
        CA --> UNET
        UNET --> DDIM[DDIM Scheduler<br/>20 denoising steps]
        DDIM --> VAE_D[VAE Decoder<br/>RGB frames]
    end
    
    subgraph "Output Processing"
        VAE_D --> BLEND[Face Blending<br/>Mask-based composition]
        BLEND --> UP[Upscaling<br/>RealESRGAN optional]
        UP --> OV[Output Video<br/>Lip-synced result]
    end
    
    subgraph "Training Components"
        SYNC[SyncNet Loss<br/>Lip-sync accuracy]
        LPIPS[LPIPS Loss<br/>Perceptual quality]
        TREPA[TREPA Loss<br/>Temporal consistency]
        L1[L1 Loss<br/>Pixel reconstruction]
    end
    
    UNET -.-> SYNC
    UNET -.-> LPIPS  
    UNET -.-> TREPA
    UNET -.-> L1
    
    classDef input fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef output fill:#e8f5e8
    classDef loss fill:#fff3e0
    
    class VI,AI input
    class FD,WE,AT,MG,VAE_E,CONCAT,CA,UNET,DDIM,VAE_D processing
    class BLEND,UP,OV output
    class SYNC,LPIPS,TREPA,L1 loss
```

### Data Flow Overview
The system processes input video and audio through a sophisticated pipeline that maintains temporal consistency while achieving high-quality lip synchronization through latent space diffusion.

### Major Components
- **UNet3D**: Core diffusion model with audio conditioning - `latentsync/models/unet.py`
- **SyncNet**: Lip-sync quality discriminator - `latentsync/models/stable_syncnet.py`
- **Audio Processing**: Whisper-based feature extraction - `latentsync/whisper/`
- **Video Pipeline**: End-to-end inference system - `latentsync/pipelines/lipsync_pipeline.py`
- **Training System**: Multi-stage training with various losses - `scripts/train_unet.py`

## 2. COMPONENT MAP

### Component Dependencies Diagram

```mermaid
flowchart LR
    subgraph "Audio Path"
        A[Audio Input] --> W[Whisper Encoder]
        W --> AE[Audio Embeddings<br/>384/768-dim]
        AE --> CA[Cross-Attention<br/>in UNet]
    end
    
    subgraph "Video Path"  
        V[Video Input] --> FD[Face Detection<br/>InsightFace]
        FD --> AT[Affine Transform]
        AT --> M[Mask Generation]
        AT --> VE[VAE Encode]
        VE --> L[Latent Space<br/>4-channel]
        M --> CC[Channel Concat<br/>13-channel]
        L --> CC
    end
    
    subgraph "Diffusion Process"
        CC --> U[UNet3D<br/>Denoising]
        CA --> U
        U --> D[DDIM Scheduler<br/>20 steps]
        D --> VD[VAE Decode]
        VD --> O[Output Frames]
    end
    
    subgraph "Training/Evaluation"
        O --> SN[SyncNet<br/>Evaluation]
        O --> LP[LPIPS Loss]
        O --> TR[TREPA Loss]
        SN --> LOSS[Combined Loss]
        LP --> LOSS
        TR --> LOSS
    end
    
    classDef audio fill:#e3f2fd
    classDef video fill:#f1f8e9  
    classDef diffusion fill:#fce4ec
    classDef training fill:#fff8e1
    
    class A,W,AE,CA audio
    class V,FD,AT,M,VE,L,CC video
    class U,D,VD,O diffusion
    class SN,LP,TR,LOSS training
```

### Entry Points
- **Gradio Interface**: `gradio_app.py:1` - Web UI for easy access
- **CLI Inference**: `scripts/inference.py:1` - Command-line interface
- **Training**: `scripts/train_unet.py:1` - UNet training entry
- **Prediction API**: `predict.py:1` - Cog deployment interface

### Configuration System
- **Audio Config**: `configs/audio.yaml` - Sample rates, mel specs, window sizes
- **UNet Configs**: `configs/unet/*.yaml` - Model architecture and training settings
- **SyncNet Configs**: `configs/syncnet/*.yaml` - Discriminator configurations
- **Scheduler**: `configs/scheduler_config.json` - DDIM inference settings

## 3. KEY PATTERNS

### Training Pipeline Diagram

```mermaid
graph TB
    subgraph "Stage 1: Base Training"
        D1[Dataset Loading<br/>5-10s segments] --> PP1[Preprocessing<br/>Face detection & affine]
        PP1 --> A1[Audio Features<br/>Whisper encoding]
        PP1 --> V1[Video Processing<br/>VAE encoding + masking]
        A1 --> U1[UNet Training<br/>L1 + LPIPS + SyncNet]
        V1 --> U1
        U1 --> C1[Checkpoint Save<br/>23-30 GB VRAM]
    end
    
    subgraph "Stage 2: Refinement"
        C1 --> D2[Fine Dataset<br/>High quality videos]
        D2 --> PP2[Enhanced Processing<br/>Better face alignment]
        PP2 --> U2[UNet Refinement<br/>Lower learning rate]
        U2 --> C2[Final Model<br/>20-55 GB VRAM]
    end
    
    subgraph "Loss Components"
        L1L[L1 Loss<br/>Pixel accuracy]
        LPIPSL[LPIPS Loss<br/>Perceptual quality]
        SYNCL[SyncNet Loss<br/>Lip-sync accuracy]
        TREPAL[TREPA Loss<br/>Temporal consistency]
    end
    
    U1 --> L1L
    U1 --> LPIPSL
    U1 --> SYNCL
    U2 --> L1L
    U2 --> LPIPSL
    U2 --> SYNCL
    U2 --> TREPAL
    
    classDef stage1 fill:#e8f5e8
    classDef stage2 fill:#e3f2fd
    classDef loss fill:#fff3e0
    
    class D1,PP1,A1,V1,U1,C1 stage1
    class D2,PP2,U2,C2 stage2
    class L1L,LPIPSL,SYNCL,TREPAL loss
```

### Model Architecture Details

```mermaid
graph TB
    subgraph "UNet3D Architecture"
        INPUT[Input<br/>13 channels] --> CONV_IN[Conv Input<br/>13→320]
        
        subgraph "Encoder"
            CONV_IN --> DOWN1[DownBlock3D<br/>320→320]
            DOWN1 --> DOWN2[CrossAttnDownBlock3D<br/>320→640]
            DOWN2 --> DOWN3[CrossAttnDownBlock3D<br/>640→1280]
            DOWN3 --> DOWN4[CrossAttnDownBlock3D<br/>1280→1280]
        end
        
        subgraph "Middle Block"
            DOWN4 --> MID[UNetMidBlock3DCrossAttn<br/>1280 channels]
        end
        
        subgraph "Decoder"
            MID --> UP1[CrossAttnUpBlock3D<br/>2560→1280]
            UP1 --> UP2[CrossAttnUpBlock3D<br/>2560→640]
            UP2 --> UP3[CrossAttnUpBlock3D<br/>1280→320]
            UP3 --> UP4[UpBlock3D<br/>640→320]
        end
        
        UP4 --> CONV_OUT[Conv Output<br/>320→4]
        CONV_OUT --> OUTPUT[Output<br/>4 channels]
        
        subgraph "Cross-Attention"
            AUDIO[Audio Features<br/>384/768-dim] --> CA1[Cross-Attention]
            CA1 --> DOWN2
            CA1 --> DOWN3
            CA1 --> DOWN4
            CA1 --> MID
            CA1 --> UP1
            CA1 --> UP2
            CA1 --> UP3
        end
    end
    
    classDef input fill:#e1f5fe
    classDef encoder fill:#f3e5f5
    classDef middle fill:#fff3e0
    classDef decoder fill:#e8f5e8
    classDef attention fill:#fce4ec
    
    class INPUT,CONV_IN input
    class DOWN1,DOWN2,DOWN3,DOWN4 encoder
    class MID middle
    class UP1,UP2,UP3,UP4,CONV_OUT,OUTPUT decoder
    class AUDIO,CA1 attention
```

### Data Processing Pipeline

```mermaid
flowchart TD
    subgraph "Preprocessing Steps"
        RAW[Raw Video Dataset] --> BROKEN[Remove Broken Videos<br/>Check file integrity]
        BROKEN --> RESAMPLE[Resample FPS & Audio<br/>25fps, 16kHz audio]
        RESAMPLE --> SCENE[Scene Detection<br/>PySceneDetect]
        SCENE --> SEGMENT[Segment Videos<br/>5-10 second clips]
        SEGMENT --> AFFINE[Affine Transform<br/>InsightFace landmarks]
        AFFINE --> SYNC_FILTER[Sync Confidence Filter<br/>SyncNet score > 3.0]
        SYNC_FILTER --> QUALITY[Visual Quality Filter<br/>hyperIQA score > 40]
        QUALITY --> FINAL[High Quality Dataset<br/>Ready for training]
    end
    
    subgraph "Quality Metrics"
        QM1[File Integrity Check]
        QM2[Frame Rate Validation]
        QM3[Scene Boundary Detection] 
        QM4[Face Detection Success]
        QM5[Audio-Visual Sync Score]
        QM6[Perceptual Quality Score]
    end
    
    BROKEN --> QM1
    RESAMPLE --> QM2
    SCENE --> QM3
    AFFINE --> QM4
    SYNC_FILTER --> QM5
    QUALITY --> QM6
    
    classDef processing fill:#f3e5f5
    classDef quality fill:#fff3e0
    classDef output fill:#e8f5e8
    
    class RAW,BROKEN,RESAMPLE,SCENE,SEGMENT,AFFINE,SYNC_FILTER,QUALITY processing
    class QM1,QM2,QM3,QM4,QM5,QM6 quality
    class FINAL output
```

### Code Patterns
- **Config Loading**: YAML-based with OmegaConf - Example: `scripts/train_unet.py`
- **Model Checkpointing**: State dict with optimizer states - Example: `scripts/train_unet.py`
- **Batch Processing**: 16-frame chunks with overlap - Example: `latentsync/pipelines/lipsync_pipeline.py`
- **Loss Combination**: Multi-objective with weights - Example: `scripts/train_unet.py`

### Common Abstractions
- **AudioEncoder**: Whisper wrapper for consistent audio features
- **FaceDetector**: Unified interface for face detection/tracking
- **VideoReader/Writer**: Abstraction over video I/O operations
- **ImageProcessor**: Face alignment and mask generation utilities

### Anti-patterns to Avoid
- Don't process video all at once - use chunking for memory efficiency
- Don't use pixel-space diffusion - stay in latent space
- Don't ignore temporal consistency - use overlapping chunks
- Don't hardcode paths - use config files

## 4. NAVIGATION GUIDE

### Quick Find
- **Main Pipeline**: `latentsync/pipelines/lipsync_pipeline.py:48` - LipsyncPipeline class
- **UNet Model**: `latentsync/models/unet.py:39` - UNet3DConditionModel
- **Audio Processing**: `latentsync/whisper/audio2feature.py:1` - Audio feature extraction
- **Face Detection**: `latentsync/utils/face_detector.py:1` - Multi-backend face detection
- **Training Script**: `scripts/train_unet.py:1` - Main training loop
- **Inference Script**: `scripts/inference.py:1` - Command-line inference

### Directory Structure Diagram

```mermaid
graph TB
    ROOT[LatentSync/] --> CORE[latentsync/<br/>Core Library]
    ROOT --> CONFIGS[configs/<br/>Configuration Files]
    ROOT --> SCRIPTS[scripts/<br/>Training & Inference]
    ROOT --> PREPROCESS[preprocess/<br/>Data Preparation]
    ROOT --> EVAL[eval/<br/>Evaluation Tools]
    ROOT --> TOOLS[tools/<br/>Utilities]
    
    CORE --> DATA[data/<br/>Dataset Loaders]
    CORE --> MODELS[models/<br/>Neural Networks]
    CORE --> PIPELINES[pipelines/<br/>Inference Logic]
    CORE --> UTILS[utils/<br/>Helper Functions]
    CORE --> WHISPER[whisper/<br/>Audio Processing]
    CORE --> TREPA[trepa/<br/>Loss Functions]
    
    CONFIGS --> AUDIO_CFG[audio.yaml]
    CONFIGS --> UNET_CFG[unet/*.yaml]
    CONFIGS --> SYNC_CFG[syncnet/*.yaml]
    
    SCRIPTS --> TRAIN[train_unet.py]
    SCRIPTS --> INFER[inference.py]
    SCRIPTS --> TRAIN_SYNC[train_syncnet.py]
    
    classDef core fill:#e3f2fd
    classDef config fill:#f1f8e9
    classDef script fill:#fce4ec
    classDef tool fill:#fff8e1
    
    class CORE,DATA,MODELS,PIPELINES,UTILS,WHISPER,TREPA core
    class CONFIGS,AUDIO_CFG,UNET_CFG,SYNC_CFG config  
    class SCRIPTS,TRAIN,INFER,TRAIN_SYNC script
    class PREPROCESS,EVAL,TOOLS tool
```

### Directory Purposes
- `/latentsync`: Core library implementation
  - `/data`: Dataset loaders for training (SyncNet & UNet datasets)
  - `/models`: Neural network architectures (UNet3D, SyncNet, attention modules)
  - `/pipelines`: Inference pipelines (LipsyncPipeline)
  - `/utils`: Helper functions (face detection, audio processing, image utilities)
  - `/whisper`: Audio feature extraction using Whisper
  - `/trepa`: Temporal consistency loss functions
- `/configs`: Configuration files for different training stages
- `/scripts`: Training and inference entry points
- `/preprocess`: Data preparation and filtering tools
- `/eval`: Evaluation metrics and benchmarking tools
- `/tools`: Utility scripts for dataset management

## 5. TASK PLAYBOOK

### Common Tasks

#### Adding a new loss function
1. Check existing losses in `scripts/train_unet.py` - search for loss computation
2. Add loss computation in training loop
3. Update loss weights in config files under `configs/unet/`
4. Add loss to tensorboard logging

#### Modifying audio processing
1. Start at `latentsync/whisper/audio2feature.py`
2. Check audio config in `configs/audio.yaml`
3. Update feature dimensions if needed
4. Ensure compatibility with UNet cross-attention

#### Changing video resolution
1. Update VAE configuration in pipeline
2. Modify face detection resolution in `latentsync/utils/face_detector.py`
3. Adjust UNet architecture if needed
4. Update configs for new resolution (stage1_512.yaml, stage2_512.yaml)

#### Debugging inference issues
1. Start at main entry points: `gradio_app.py` or `scripts/inference.py`
2. Check logs in `latentsync/pipelines/lipsync_pipeline.py`
3. Common issues:
   - **OOM**: Reduce batch size, enable gradient checkpointing, use efficient configs
   - **Face detection failure**: Check input video quality, try different detection backends
   - **Audio sync issues**: Verify audio sample rate (16kHz), check Whisper model size
   - **Quality issues**: Adjust guidance_scale (1.0-3.0), increase inference_steps (20-50)

#### Training from scratch
1. Prepare dataset using tools in `preprocess/` directory
2. Run complete data processing pipeline: `./data_processing_pipeline.sh`
3. Create file lists for train/val splits using `tools/write_fileslist.py`
4. Copy and modify config from `configs/unet/` (start with stage1.yaml)
5. Start training: `./train_unet.sh`
6. Monitor progress with tensorboard
7. Switch to stage2 config for refinement training

#### Fine-tuning on custom data
1. Use `fine_tuning.sh` as template
2. Prepare smaller dataset (100-1000 samples)
3. Use lower learning rate (1e-6 to 5e-6)
4. Train for fewer steps (5000-10000)
5. Monitor for overfitting

### Performance Optimization Strategies

#### Memory Optimization
- **Training**: Use gradient checkpointing, mixed precision (FP16), efficient configs
- **Inference**: Reduce batch size to 1, enable CPU offloading for large models
- **Data Loading**: Use smaller resolution configs (256x256 vs 512x512)

#### Speed Optimization  
- **Training**: Enable xformers attention, use compiled models, optimize data loading
- **Inference**: Reduce DDIM steps (10-15), use efficient schedulers, batch processing
- **Deployment**: Use TensorRT for production, quantization for edge devices

#### Quality Enhancement
- **Training**: Use higher resolution configs, increase training steps, better data quality
- **Inference**: Increase guidance scale (2.0-3.0), more denoising steps (30-50)
- **Post-processing**: Enable upscaling with RealESRGAN, better face blending

### Model Deployment Options

#### Docker Deployment
```bash
# Build container
docker build -t latentsync .

# Run with GPU support
docker run --gpus all -p 7860:7860 latentsync
```

#### API Deployment
- **Cog**: Use `predict.py` for Replicate deployment
- **Gradio**: Built-in API endpoints at `/api/v1/`  
- **Custom**: Wrap `LipsyncPipeline` in FastAPI/Flask

#### Batch Processing
```bash
# Process multiple videos
python scripts/inference.py --input_dir /path/to/videos --output_dir /path/to/results
```

## 6. TECHNICAL SPECIFICATIONS

### Model Versions
- **LatentSync 1.5**: 256x256 resolution, temporal consistency improvements, 8GB VRAM minimum
- **LatentSync 1.6**: 512x512 resolution, reduced blurriness, 18GB VRAM minimum

### Hardware Requirements
- **Training**: 20-55GB VRAM, 32GB+ RAM, fast NVMe storage
- **Inference**: 8-18GB VRAM (depending on version), 16GB RAM
- **Batch Processing**: Scale linearly with number of concurrent videos

### Supported Formats
- **Input**: MP4, AVI, MOV (video), WAV, MP3, AAC (audio)
- **Output**: MP4 with H.264 encoding, configurable quality settings
- **Intermediate**: PyTorch tensors, cached embeddings (.pt files)

This documentation provides comprehensive coverage of the LatentSync architecture with detailed Mermaid diagrams for better understanding and navigation of the complex system.