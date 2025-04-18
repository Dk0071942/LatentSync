# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..utils.image_upscale import ImageUpscale
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

import tempfile

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        denoising_unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(denoising_unet.config, "_diffusers_version") and version.parse(
            version.parse(denoising_unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(denoising_unet.config, "sample_size") and denoising_unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(denoising_unet.config)
            new_config["sample_size"] = 64
            denoising_unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

        self.image_upscaler = ImageUpscale()

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.denoising_unet, "_hf_hook"):
            return self.device
        for module in self.denoising_unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        face_detected_flags = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            result = self.image_processor.affine_transform(frame)
            if len(result) == 4:
                face, box, affine_matrix, face_detected = result
            else:
                face, box, affine_matrix = result
                face_detected = face is not None

            if face_detected:
                faces.append(face)
                boxes.append(box)
                affine_matrices.append(affine_matrix)
                face_detected_flags.append(True)
            else:
                faces.append(None)
                boxes.append(None)
                affine_matrices.append(None)
                face_detected_flags.append(False)

        valid_faces = [f for f in faces if f is not None]
        if valid_faces:
            pass

        return faces, boxes, affine_matrices, face_detected_flags

    def restore_video(self, faces: Optional[torch.Tensor], video_frames: np.ndarray, boxes: list, affine_matrices: list, face_detected_flags: list) -> np.ndarray:
        num_frames_to_process = len(video_frames) # Process all original frames

        if num_frames_to_process == 0:
            return np.array([], dtype=np.uint8)

        # Determine output shape and dtype from the first original frame
        output_dtype = video_frames[0].dtype
        output_shape = video_frames.shape

        temp_fd, temp_path = tempfile.mkstemp(suffix='.mmap', prefix='video_restore_')
        os.close(temp_fd)

        output_memmap = None
        processed_frame_idx = 0 # Index for the 'faces' tensor
        try:
            output_memmap = np.memmap(temp_path, dtype=output_dtype, mode='w+', shape=output_shape)

            print(f"Restoring {num_frames_to_process} frames...")
            for index in tqdm.tqdm(range(num_frames_to_process)):
                if face_detected_flags[index]:
                    # Ensure we have processed faces data and corresponding box/matrix
                    if faces is None or processed_frame_idx >= len(faces) or boxes[index] is None or affine_matrices[index] is None:
                         print(f"Warning: Face detected flag is True for frame {index}, but required data is missing. Using original frame.")
                         output_memmap[index] = video_frames[index]
                         # Do not increment processed_frame_idx if we didn't use a processed face
                         continue # Skip to next frame

                    face_tensor = faces[processed_frame_idx]
                    x1, y1, x2, y2 = boxes[index]
                    h_face = int(y2 - y1)
                    w_face = int(x2 - x1)

                    # Resize the processed face tensor to the original detected box size
                    face_tensor_resized = torchvision.transforms.functional.resize(face_tensor, size=(h_face, w_face), antialias=True)
                    face_tensor_resized = rearrange(face_tensor_resized, "c h w -> h w c")
                    face_tensor_resized = (face_tensor_resized / 2 + 0.5).clamp(0, 1)
                    face_np = (face_tensor_resized * 255).to(torch.uint8).cpu().numpy()

                    current_video_frame = video_frames[index]

                    # Restore the face into the original frame
                    out_frame = self.image_processor.restorer.restore_img(
                        current_video_frame, face_np, affine_matrices[index]
                    )
                    output_memmap[index] = out_frame
                    processed_frame_idx += 1 # Move to the next processed face
                else:
                    # No face detected, use the original frame directly
                    output_memmap[index] = video_frames[index]

            output_memmap.flush()
            final_ram_array = np.array(output_memmap)
            return final_ram_array

        except Exception as e:
            # Include index in error for better debugging
            print(f"Error during video restoration at frame index {index}: {e}")
            raise e
        finally:
            if output_memmap is not None:
                if hasattr(output_memmap, '_mmap'):
                    try:
                        output_memmap._mmap.close()
                    except Exception:
                        pass
                del output_memmap

            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                     print(f"Error removing temporary file {temp_path}: {e}") # Log error

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            # Pass the flags list as well
            faces, boxes, affine_matrices, face_detected_flags = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_face_detected_flags = [] # Initialize list for flags
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces += faces # Use extend for lists
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_face_detected_flags += face_detected_flags # Add flags
                else:
                    loop_video_frames.append(video_frames[::-1])
                    # Reverse lists correctly, handle potential None
                    loop_faces += faces[::-1]
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_face_detected_flags += face_detected_flags[::-1] # Add reversed flags

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            # Slice the lists
            faces = loop_faces[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            face_detected_flags = loop_face_detected_flags[: len(whisper_chunks)] # Slice flags
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            # Get flags here too
            faces, boxes, affine_matrices, face_detected_flags = self.affine_transform_video(video_frames)

        # Return the flags list
        return video_frames, faces, boxes, affine_matrices, face_detected_flags

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        mask_image_path: str = "latentsync/utils/mask.png",
        enable_upscale: Optional[bool] = True,
        sharpness_factor: Optional[float] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.denoising_unet.training
        self.denoising_unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.denoising_unet.config.sample_size * self.vae_scale_factor
        width = width or self.denoising_unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        video_frames, faces, boxes, affine_matrices, face_detected_flags = self.loop_video(whisper_chunks, video_frames)

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            batch_size,
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.denoising_unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None

            # Get current chunk of faces and flags
            current_chunk_faces = faces[i * num_frames : (i + 1) * num_frames]
            current_chunk_flags = face_detected_flags[i * num_frames : (i + 1) * num_frames]

            # Filter for valid faces in the chunk
            valid_faces_in_chunk = [face for face, flag in zip(current_chunk_faces, current_chunk_flags) if flag]

            # If no faces detected in this chunk, skip processing and add placeholder
            if not valid_faces_in_chunk:
                synced_video_frames.append(None) # Placeholder for restore_video
                # masked_video_frames.append(None)
                continue # Skip to the next chunk

            # Stack valid faces into a tensor
            inference_faces_tensor = torch.stack(valid_faces_in_chunk)

            # Get the main latents for the full chunk size
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]

            # Prepare masks and images using the *tensor* of valid faces
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces_tensor, affine_transform=False # Pass the tensor
            )

            # 7. Prepare mask latent variables - these will correspond to valid faces only
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks, # Should have shape matching valid faces
                masked_pixel_values, # Should have shape matching valid faces
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents - these will correspond to valid faces only
            ref_latents = self.prepare_image_latents(
                ref_pixel_values, # Should have shape matching valid faces
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # Select latents corresponding to valid frames ONLY for denoising input
            # We need the indices of valid frames within the chunk
            valid_indices_in_chunk = [idx for idx, flag in enumerate(current_chunk_flags) if flag]
            # Ensure latents match the number of valid faces for concatenation
            valid_latents = latents[:, :, valid_indices_in_chunk, :, :] # Select frames

            # 9. Denoising loop - operates on latents corresponding to valid frames
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    # Use valid_latents which matches the number of valid faces
                    denoising_unet_input = torch.cat([valid_latents] * 2) if do_classifier_free_guidance else valid_latents

                    denoising_unet_input = self.scheduler.scale_model_input(denoising_unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    # Ensure dimensions match: valid_latents, mask_latents, masked_image_latents, ref_latents
                    # All should have frame dimension = len(valid_faces_in_chunk)
                    denoising_unet_input = torch.cat(
                        [denoising_unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                    )

                    # predict the noise residual
                    # Audio embeds might need adjustment if shape depends on frame count?
                    # Assuming audio_embeds shape is independent or handled by UNet
                    noise_pred = self.denoising_unet(
                        denoising_unet_input, t, encoder_hidden_states=audio_embeds # Pass potentially duplicated audio embeds
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    # Update valid_latents
                    valid_latents = self.scheduler.step(noise_pred, t, valid_latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, valid_latents) # Pass valid_latents to callback

            # Recover the pixel values for the processed frames
            # Use valid_latents which contains the final denoised latents for valid frames
            decoded_latents = self.decode_latents(valid_latents)
            if enable_upscale:
                decoded_latents = self.image_upscaler.enhance_and_upscale(decoded_latents, sharpness_factor)
            # Paste back using masks derived from valid faces
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            # Append the result for this chunk (contains only processed frames)
            synced_video_frames.append(decoded_latents)
            # masked_video_frames.append(masked_pixel_values) # This would only contain masks for valid frames

        # Filter out None placeholders before concatenating for restore_video
        valid_synced_frames = [f for f in synced_video_frames if f is not None]

        if not valid_synced_frames:
            print("Warning: No faces were detected or processed in any frame.")
            # If no frames were processed at all, return the original video unchanged (or handle as error)
            # Need to decide the desired behavior here. Returning original seems reasonable.
            # However, restore_video expects a tensor or None. Let's pass None.
            concatenated_frames = None
        else:
            # Concatenate results from chunks that had valid faces
            concatenated_frames = torch.cat(valid_synced_frames)

        # Pass potentially None concatenated_frames if no valid frames were processed across all chunks
        synced_video_frames = self.restore_video(concatenated_frames, video_frames, boxes, affine_matrices, face_detected_flags)

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.denoising_unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=25)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
