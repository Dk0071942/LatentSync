# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import argparse
import shutil
import datetime
import logging
from omegaconf import OmegaConf

from tqdm.auto import tqdm
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.logging import get_logger
from diffusers.optimization import get_scheduler
from accelerate.utils import set_seed

from latentsync.data.unet_dataset import UNetDataset
from latentsync.models.unet import UNet3DConditionModel
from latentsync.models.stable_syncnet import StableSyncNet
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.util import (
    init_dist,
    cosine_loss,
    one_step_sampling,
    validation
)
from latentsync.utils.util import plot_loss_chart
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.trepa.loss import TREPALoss
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from eval.eval_sync_conf import syncnet_eval
import lpips


logger = get_logger(__name__)


def main(config):
    # Initialize distributed training
    local_rank = init_dist()
    is_distributed = local_rank != -1

    if is_distributed:
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        is_main_process = global_rank == 0
    else:
        # When not in distributed mode, we are the only process.
        local_rank = 0
        global_rank = 0
        num_processes = 1
        is_main_process = True

    seed = config.run.seed + global_rank
    set_seed(seed)

    # Logging folder
    folder_name = "train" + datetime.datetime.now().strftime("-%Y_%m_%d-%H_%M_%S")
    output_dir = os.path.join(config.data.train_output_dir, folder_name)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        diffusers.utils.logging.set_verbosity_info()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/val_videos", exist_ok=True)
        os.makedirs(f"{output_dir}/sync_conf_results", exist_ok=True)
        shutil.copy(config.unet_config_path, output_dir)
        shutil.copy(config.data.syncnet_config_path, output_dir)

    if is_distributed:
        device = torch.device(local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_scheduler = DDIMScheduler.from_pretrained("configs")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae.requires_grad_(False)
    vae.to(device)

    if config.run.pixel_space_supervise:
        vae.enable_gradient_checkpointing()

    syncnet_eval_model = SyncNetEval(device=device)
    syncnet_eval_model.loadParameters("checkpoints/auxiliary/syncnet_v2.model")

    syncnet_detector = SyncNetDetector(device=device, detect_results_dir="detect_results")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        audio_embeds_cache_dir=config.data.audio_embeds_cache_dir,
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    unet, resume_global_step = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        config.ckpt.resume_ckpt_path,
        device=device,
    )

    if config.model.add_audio_layer and config.run.use_syncnet:
        syncnet_config = OmegaConf.load(config.data.syncnet_config_path)
        if syncnet_config.ckpt.inference_ckpt_path == "":
            raise ValueError("SyncNet path is not provided")
        syncnet = StableSyncNet(OmegaConf.to_container(syncnet_config.model), gradient_checkpointing=True).to(
            device=device, dtype=torch.float32
        )
        syncnet_checkpoint = torch.load(
            syncnet_config.ckpt.inference_ckpt_path, map_location=device, weights_only=True
        )
        syncnet.load_state_dict(syncnet_checkpoint["state_dict"])
        syncnet.requires_grad_(False)

        del syncnet_checkpoint
        torch.cuda.empty_cache()

    if config.model.use_motion_module:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            for trainable_module_name in config.run.trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    else:
        unet.requires_grad_(True)
        trainable_params = list(unet.parameters())

    if config.optimizer.scale_lr:
        config.optimizer.lr = config.optimizer.lr * num_processes

    optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)

    if is_main_process:
        logger.info(f"trainable params number: {len(trainable_params)}")
        logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable gradient checkpointing
    if config.run.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Get the training dataset
    train_dataset = UNetDataset(config.data.train_data_dir, config)
    
    if is_distributed:
        distributed_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_processes,
            rank=global_rank,
            shuffle=True,
            seed=config.run.seed,
        )
    else:
        distributed_sampler = None

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=False if distributed_sampler else True,
        sampler=distributed_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    # Get the training iteration
    if config.run.max_train_steps == -1:
        assert config.run.max_train_epochs != -1
        config.run.max_train_steps = config.run.max_train_epochs * len(train_dataloader)

    # Scheduler
    lr_scheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps,
        num_training_steps=config.run.max_train_steps,
    )

    if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
        lpips_loss_func = lpips.LPIPS(net="vgg").to(device)

    if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
        trepa_loss_func = TREPALoss(device=device, with_cp=True)

    # Validation pipeline
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=noise_scheduler,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # DDP wrapper
    if is_distributed:
        unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(config.run.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = config.data.batch_size * num_processes

    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.data.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {config.run.max_train_steps}")
    global_step = resume_global_step
    first_epoch = resume_global_step // num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, config.run.max_train_steps),
        initial=resume_global_step,
        desc="Steps",
        disable=not is_main_process,
    )

    loss_list = []
    recon_loss_list = []
    perceptual_loss_list = []
    trepa_loss_list = []
    sync_loss_list = []

    val_loss_list = []
    sync_conf_list = []
    val_step_list = []

    for epoch in range(first_epoch, num_train_epochs):
        if is_distributed:
            distributed_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            unet.train()
            # ########################################################################################################
            # #                                               forward                                                #
            # ########################################################################################################
            video_frames = batch["video_frames"].to(device, dtype=torch.float16)
            audio_feat = batch["audio_feat"].to(device, dtype=torch.float16)
            gt_sync_net_hidden = batch["gt_sync_net_hidden"].to(device, dtype=torch.float16)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(video_frames)
            bsz = video_frames.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(video_frames, noise, timesteps)

            # Predict the noise residual
            pred_noise, additional_hidden_states = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=audio_feat,
                return_dict=False,
            )

            # ########################################################################################################
            # #                                                loss                                                  #
            # ########################################################################################################
            loss = F.mse_loss(pred_noise, noise, reduction="mean")

            if config.run.pixel_space_supervise:
                pred_original_sample = one_step_sampling(noise_scheduler, pred_noise, timesteps, noisy_latents)
                pred_video_frames = vae.decode(pred_original_sample / vae.config.scaling_factor).sample

                # l2 loss
                recon_loss = F.mse_loss(pred_video_frames, video_frames) * config.run.recon_loss_weight
                loss += recon_loss

                if config.run.perceptual_loss_weight != 0:
                    perceptual_loss = (
                        lpips_loss_func(pred_video_frames, video_frames).mean() * config.run.perceptual_loss_weight
                    )
                    loss += perceptual_loss
                else:
                    perceptual_loss = None

                if config.run.trepa_loss_weight != 0:
                    trepa_loss = trepa_loss_func(pred_video_frames, video_frames) * config.run.trepa_loss_weight
                    loss += trepa_loss
                else:
                    trepa_loss = None
            else:
                recon_loss = None
                perceptual_loss = None
                trepa_loss = None

            if config.model.add_audio_layer and config.run.use_syncnet:
                # audio-visual sync loss
                pred_sync_net_hidden = syncnet.forward(
                    pred_video_frames, audio_feat, unet_hidden_states=additional_hidden_states
                )
                sync_loss = (
                    cosine_loss(pred_sync_net_hidden, gt_sync_net_hidden, y=torch.ones(len(pred_sync_net_hidden)).to(device))
                    * config.run.sync_loss_weight
                )
                loss += sync_loss.mean()
                sync_loss = sync_loss.mean()
            else:
                sync_loss = None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.optimizer.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Gather the losses across all processes for logging (if we use distributed training).
            if is_distributed:
                avg_loss = gather_loss(loss, device)
                avg_recon_loss = gather_loss(recon_loss, device) if recon_loss is not None else 0.0
                avg_perceptual_loss = gather_loss(perceptual_loss, device) if perceptual_loss is not None else 0.0
                avg_trepa_loss = gather_loss(trepa_loss, device) if trepa_loss is not None else 0.0
                avg_sync_loss = gather_loss(sync_loss, device) if sync_loss is not None else 0.0
            else:
                avg_loss = loss.item()
                avg_recon_loss = recon_loss.item() if recon_loss is not None else 0.0
                avg_perceptual_loss = perceptual_loss.item() if perceptual_loss is not None else 0.0
                avg_trepa_loss = trepa_loss.item() if trepa_loss is not None else 0.0
                avg_sync_loss = sync_loss.item() if sync_loss is not None else 0.0

            if is_main_process:
                loss_list.append((global_step, avg_loss))
                if config.run.pixel_space_supervise:
                    recon_loss_list.append((global_step, avg_recon_loss))
                    perceptual_loss_list.append((global_step, avg_perceptual_loss))
                    trepa_loss_list.append((global_step, avg_trepa_loss))
                if config.run.use_syncnet:
                    sync_loss_list.append((global_step, avg_sync_loss))

                logs = {"step_loss": avg_loss, "epoch": epoch}
                progress_bar.set_postfix(**logs)

            progress_bar.update(1)
            global_step += 1

            if is_main_process:
                # periodic validation
                if global_step % config.run.validation_steps == 0 or global_step == 1:
                    val_step_list.append(global_step)
                    sync_conf, val_loss = validation(
                        config=config,
                        pipeline=pipeline,
                        device=device,
                        epoch=epoch,
                        step=global_step,
                        output_dir=output_dir,
                        syncnet_eval_model=syncnet_eval_model,
                        syncnet_detector=syncnet_detector,
                    )
                    val_loss_list.append(val_loss)
                    sync_conf_list.append(sync_conf)
                    plot_loss_chart(f"{output_dir}/val_loss_curve.png", ("val_loss", val_step_list, val_loss_list))
                    plot_loss_chart(
                        f"{output_dir}/sync_conf_curve.png", ("sync_conf", val_step_list, sync_conf_list)
                    )

                # periodic saving
                if global_step % config.run.checkpointing_steps == 0:
                    save_path = os.path.join(output_dir, "checkpoints", f"unet_step_{global_step}.pt")
                    if is_distributed:
                        torch.save(unet.module.state_dict(), save_path)
                    else:
                        torch.save(unet.state_dict(), save_path)
                    logger.info(f"Saved state to {save_path} (global_step: {global_step})")
                plot_loss_chart(
                    f"{output_dir}/train_loss_curve.png",
                    ("total_loss", [item[0] for item in loss_list], [item[1] for item in loss_list]),
                    (
                        "recon_loss",
                        [item[0] for item in recon_loss_list],
                        [item[1] for item in recon_loss_list],
                    ),
                    (
                        "perceptual_loss",
                        [item[0] for item in perceptual_loss_list],
                        [item[1] for item in perceptual_loss_list],
                    ),
                    ("trepa_loss", [item[0] for item in trepa_loss_list], [item[1] for item in trepa_loss_list]),
                    ("sync_loss", [item[0] for item in sync_loss_list], [item[1] for item in sync_loss_list]),
                )

        if global_step >= config.run.max_train_steps:
            break

    progress_bar.close()
    if is_distributed:
        dist.destroy_process_group()


def gather_loss(loss, device):
    # Sum the local loss across all processes
    local_loss = loss.item()
    global_loss = torch.tensor(local_loss, dtype=torch.float32).to(device)
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)

    # Calculate the average loss across all processes
    global_average_loss = global_loss.item() / dist.get_world_size()
    return global_average_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/stage2.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.unet_config_path)
    config.unet_config_path = args.unet_config_path

    main(config)
