# LatentSync Documentation Index

This index helps you navigate the LatentSync project documentation efficiently.

## 📖 Main Documentation Files

### Core Documentation
- **[README.md](README.md)** - Project overview, setup, and basic usage
- **[PROJECT_DOCS.md](PROJECT_DOCS.md)** - **⭐ RECOMMENDED** - Comprehensive technical documentation with architectural diagrams
- **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)** - In-depth technical analysis, bottlenecks, and optimization opportunities
- **[ECCV2022-RIFE/ECCV2022-RIFE_DOCS.md](ECCV2022-RIFE/ECCV2022-RIFE_DOCS.md)** - **🆕 NEW** - Complete RIFE frame interpolation documentation with diagrams

### Configuration & Setup
- **[setup_env.sh](setup_env.sh)** - Environment setup script
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[cog.yaml](cog.yaml)** - Replicate deployment configuration
- **[Dockerfile](Dockerfile)** - Docker container setup

### Training & Scripts
- **[train_unet.sh](train_unet.sh)** - UNet training script
- **[train_syncnet.sh](train_syncnet.sh)** - SyncNet training script
- **[inference.sh](inference.sh)** - Inference execution script
- **[fine_tuning.sh](fine_tuning.sh)** - Fine-tuning workflow
- **[data_processing_pipeline.sh](data_processing_pipeline.sh)** - Data preparation pipeline

## 🎯 Quick Access by Use Case

### 🔍 **Understanding the System**
Start here if you want to understand how LatentSync works:
1. **[PROJECT_DOCS.md](PROJECT_DOCS.md)** - Architecture overview with diagrams
2. **[ECCV2022-RIFE/ECCV2022-RIFE_DOCS.md](ECCV2022-RIFE/ECCV2022-RIFE_DOCS.md)** - RIFE frame interpolation system
3. **[docs/framework.png](docs/framework.png)** - Visual framework diagram
4. **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)** - Technical deep dive

### 🚀 **Getting Started**
Follow this path for setup and first run:
1. **[README.md](README.md)** - Basic setup instructions
2. **[setup_env.sh](setup_env.sh)** - Run this script
3. **[gradio_app.py](gradio_app.py)** - Launch web interface
4. **[inference.sh](inference.sh)** - Command-line usage

### 🏋️ **Training & Fine-tuning**
For training your own models:
1. **[PROJECT_DOCS.md#training-pipeline-diagram](PROJECT_DOCS.md#training-pipeline-diagram)** - Training overview
2. **[data_processing_pipeline.sh](data_processing_pipeline.sh)** - Prepare your data
3. **[configs/unet/](configs/unet/)** - Training configurations
4. **[train_unet.sh](train_unet.sh)** - Start training

### 🔧 **Development & Customization**
For developers wanting to modify the system:
1. **[PROJECT_DOCS.md#navigation-guide](PROJECT_DOCS.md#navigation-guide)** - Code navigation
2. **[latentsync/](latentsync/)** - Core library structure
3. **[TECHNICAL_DETAILS.md#potential-improvements](TECHNICAL_DETAILS.md#potential-improvements)** - Optimization opportunities

### 📊 **Evaluation & Benchmarking**
For testing and evaluation:
1. **[eval/](eval/)** - Evaluation scripts directory
2. **[eval/eval_sync_conf.sh](eval/eval_sync_conf.sh)** - Sync confidence evaluation
3. **[eval/eval_syncnet_acc.sh](eval/eval_syncnet_acc.sh)** - SyncNet accuracy testing

### 🎬 **Frame Interpolation with RIFE**
For video frame rate enhancement:
1. **[ECCV2022-RIFE/ECCV2022-RIFE_DOCS.md](ECCV2022-RIFE/ECCV2022-RIFE_DOCS.md)** - Complete RIFE documentation
2. **[ECCV2022-RIFE/inference_video.py](ECCV2022-RIFE/inference_video.py)** - CLI video interpolation
3. **[ECCV2022-RIFE/rife_app/](ECCV2022-RIFE/rife_app/)** - Gradio integration
4. **[gradio_app.py](gradio_app.py)** - Combined LatentSync + RIFE interface

## 📁 Directory Structure Quick Reference

```
LatentSync/
├── 📄 Core Documentation
│   ├── README.md                    # Main project overview
│   ├── PROJECT_DOCS.md     # ⭐ Complete technical docs with diagrams
│   ├── TECHNICAL_DETAILS.md        # Performance analysis & optimizations
│   └── DOCUMENTATION_INDEX.md      # This file
├── 🏗️ Core Library
│   └── latentsync/                  # Main codebase
│       ├── models/                  # Neural network architectures
│       ├── pipelines/              # Inference pipelines
│       ├── utils/                  # Utility functions
│       └── whisper/                # Audio processing
├── 🎬 RIFE Integration
│   └── ECCV2022-RIFE/             # Frame interpolation system
│       ├── ECCV2022-RIFE_DOCS.md  # 🆕 Complete RIFE documentation
│       ├── RIFE_INTEGRATION_SUMMARY.md  # Integration overview
│       ├── model/                  # IFNet architecture
│       ├── rife_app/              # LatentSync integration
│       ├── benchmark/             # Evaluation scripts
│       └── inference_*.py         # CLI tools
├── ⚙️ Configuration
│   └── configs/                     # YAML configuration files
│       ├── unet/                   # UNet training configs
│       ├── syncnet/                # SyncNet training configs
│       └── audio.yaml              # Audio processing settings
├── 🔧 Scripts & Tools
│   ├── scripts/                    # Training & inference scripts
│   ├── preprocess/                 # Data preparation tools
│   ├── eval/                       # Evaluation tools
│   └── tools/                      # Utility scripts
└── 📊 Assets & Results
    ├── assets/                     # Demo assets
    ├── results/                    # Generated outputs
    └── checkpoints/               # Model weights
```

## 🎨 Mermaid Diagrams Available

Our comprehensive documentation includes several Mermaid diagrams:

1. **System Architecture Diagram** - Complete pipeline overview
2. **Component Dependencies** - Data flow between modules  
3. **Training Pipeline** - Stage 1 & Stage 2 training process
4. **UNet3D Architecture** - Detailed model structure
5. **Data Processing Pipeline** - Preprocessing workflow
6. **Directory Structure** - Visual project organization

Find all diagrams in **[PROJECT_DOCS.md](PROJECT_DOCS.md)**.

## 🔗 External Resources

- **[Paper](https://arxiv.org/abs/2412.09262)** - LatentSync research paper
- **[HuggingFace Model](https://huggingface.co/ByteDance/LatentSync-1.6)** - Pre-trained models
- **[HuggingFace Space](https://huggingface.co/spaces/fffiloni/LatentSync)** - Online demo
- **[Replicate](https://replicate.com/lucataco/latentsync)** - Cloud API

## 📞 Getting Help

1. **Understanding Architecture**: Read **[PROJECT_DOCS.md](PROJECT_DOCS.md)**
2. **Setup Issues**: Check **[README.md](README.md)** and **[setup_env.sh](setup_env.sh)**
3. **Performance Problems**: See **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)**
4. **Training Questions**: Review **[PROJECT_DOCS.md#task-playbook](PROJECT_DOCS.md#task-playbook)**

---
*This index was generated to help navigate the LatentSync project efficiently. For the most comprehensive technical information with visual diagrams, see **[PROJECT_DOCS.md](PROJECT_DOCS.md)**.*