````markdown
# Project Commands (Windows)

## Data Processing
```cmd
python -m preprocess.data_processing_pipeline --total_num_workers 12 --per_gpu_num_workers 12 --resolution 256 --sync_conf_threshold 3 --temp_dir temp --input_dir ./training_materials
````

## Train SyncNet

```cmd
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25678 -m scripts.train_syncnet --config_path "configs/syncnet/syncnet_16_pixel_attn.yaml"
```

## Train UNet

```cmd
$env:PYTHONPATH="X:\Github_repo\LatentSync"; python scripts/train_unet.py --unet_config_path "configs/unet/stage2_efficient.yaml"
```

## Inference

```cmd
python -m scripts.inference --unet_config_path "configs/unet/stage2.yaml" --inference_ckpt_path "checkpoints/default_unet_v1.5.pt" --inference_steps 20 --guidance_scale 2.0 --video_path "assets/demo1_video.mp4" --audio_path "assets/demo1_audio.wav" --video_out_path "video_out.mp4"
```