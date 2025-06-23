# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess
import torch.nn.functional as F

import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version
import time

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
        unet: UNet3DConditionModel,
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

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
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
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
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
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
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

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
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
        """
        Restores faces in video frames using processed face tensors.

        Handles all original frames, using processed faces where available
        and original frames otherwise. Uses memmap for memory efficiency.
        Incorporates BICUBIC resizing and passes tensors directly to the restorer.
        """
        num_frames_to_process = len(video_frames) # Process all original frames

        if num_frames_to_process == 0:
            return np.array([], dtype=np.uint8)

        # Determine output shape and dtype from the first original frame
        # Ensure video_frames is not empty before accessing index 0
        if len(video_frames) > 0:
            output_dtype = video_frames[0].dtype
            output_shape = video_frames.shape
        else: # Handle edge case of empty video_frames input more robustly
             return np.array([], dtype=np.uint8)


        temp_fd, temp_path = tempfile.mkstemp(suffix='.mmap', prefix='video_restore_')
        os.close(temp_fd)

        output_memmap = None
        processed_frame_idx = 0 # Index for the 'faces' tensor
        index = -1 # Initialize index for potential use in error message outside loop

        try:
            output_memmap = np.memmap(temp_path, dtype=output_dtype, mode='w+', shape=output_shape)

            print(f"Restoring {num_frames_to_process} frames...")
            for index in tqdm.tqdm(range(num_frames_to_process)):
                if face_detected_flags[index]:
                    # Check if we have the necessary data for this flagged frame
                    # Note: Check faces is not None *and* check processed_frame_idx is within bounds
                    if faces is None or processed_frame_idx >= len(faces) or boxes[index] is None or affine_matrices[index] is None:
                        print(f"Warning: Face detected flag is True for frame {index}, but required data (face tensor/box/matrix) is missing or index mismatch. Using original frame.")
                        output_memmap[index] = video_frames[index]
                        # Do *not* increment processed_frame_idx if we didn't use a processed face
                        continue # Skip to next frame

                    # --- Start: Face processing logic from the second version ---
                    face_tensor = faces[processed_frame_idx]
                    x1, y1, x2, y2 = boxes[index]
                    # Use integer conversion robustly
                    h_face = int(round(y2 - y1))
                    w_face = int(round(x2 - x1))

                    # Ensure height and width are positive
                    if h_face <= 0 or w_face <= 0:
                         print(f"Warning: Invalid bounding box dimensions ({w_face}x{h_face}) for frame {index}. Using original frame.")
                         output_memmap[index] = video_frames[index]
                         processed_frame_idx += 1 # Increment index as we consumed the face tensor, even if we couldn't use it
                         continue


                    # Resize using BICUBIC interpolation (from the second version)
                    # Ensure face_tensor is in C, H, W format if needed by resize
                    face_tensor_resized = torchvision.transforms.functional.resize(
                        face_tensor,
                        size=(h_face, w_face),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True
                    )

                    current_video_frame = video_frames[index]

                    # Restore the face into the original frame
                    # Pass the resized *tensor* directly (from the second version)
                    # Ensure self.image_processor.restorer.restore_img can handle tensors
                    out_frame = self.image_processor.restorer.restore_img(
                        current_video_frame, face_tensor_resized, affine_matrices[index]
                    )
                    # --- End: Face processing logic from the second version ---

                    output_memmap[index] = out_frame
                    processed_frame_idx += 1 # Move to the next processed face

                else:
                    # No face detected flag, use the original frame directly
                    output_memmap[index] = video_frames[index]

            # --- Cleanup logic from the first version ---
            output_memmap.flush()
            # Copy data from memmap to an in-memory array before returning
            final_ram_array = np.array(output_memmap)
            return final_ram_array

        except Exception as e:
            # Include index in error for better debugging
            print(f"Error during video restoration at frame index {index}: {e}")
            # It might be useful to log the traceback too:
            # import traceback
            # print(traceback.format_exc())
            raise e # Re-raise the exception after logging
        finally:
            # Ensure memmap is closed and file is deleted even if errors occur
            if output_memmap is not None:
                # Check if the underlying mmap object exists and close it
                if hasattr(output_memmap, '_mmap') and output_memmap._mmap is not None:
                    try:
                        output_memmap._mmap.close()
                    except Exception as close_err:
                         print(f"Warning: Exception while closing memmap file handle: {close_err}")
                # Explicitly delete the reference to potentially trigger GC and release file lock sooner
                del output_memmap

            # Ensure the temporary file is removed
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
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        enable_upscale: Optional[bool] = True,
        sharpness_factor: Optional[float] = 1,
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        call_start_time = time.time()
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        setup_start_time = time.time()
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

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
        setup_end_time = time.time()
        print(f"Pipeline setup time: {setup_end_time - setup_start_time:.2f} seconds")

        data_prepare_start_time = time.time()
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)
        data_prepare_end_time = time.time()
        print(f"Audio/Video loading and feature extraction time: {data_prepare_end_time - data_prepare_start_time:.2f} seconds")

        loop_video_start_time = time.time()
        video_frames, faces, boxes, affine_matrices, face_detected_flags = self.loop_video(whisper_chunks, video_frames)
        loop_video_end_time = time.time()
        print(f"Video looping and face detection/affine transform time: {loop_video_end_time - loop_video_start_time:.2f} seconds")


        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        prepare_latents_start_time = time.time()
        # Prepare latent variables
        all_latents = self.prepare_latents(
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )
        prepare_latents_end_time = time.time()
        print(f"Initial latents preparation time: {prepare_latents_end_time - prepare_latents_start_time:.2f} seconds")

        inference_loop_start_time = time.time()
        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                
                # Get current chunk of faces and flags
                current_chunk_faces = faces[i * num_frames : (i + 1) * num_frames]
                current_chunk_flags = face_detected_flags[i * num_frames : (i + 1) * num_frames]
                
                # Filter for valid faces in the chunk
                valid_faces_in_chunk = [face for face, flag in zip(current_chunk_faces, current_chunk_flags) if flag]
                
                # If we have valid faces, ensure audio features match their count
                if valid_faces_in_chunk:
                    if len(valid_faces_in_chunk) != audio_embeds.shape[0]:
                        # audio_embeds shape: (frames, seq_len, features)
                        # permute to (seq_len, features, frames) to interpolate over frames.
                        audio_embeds_permuted = audio_embeds.permute(1, 2, 0)
                        audio_embeds_interpolated = F.interpolate(
                            audio_embeds_permuted,
                            size=len(valid_faces_in_chunk),
                            mode="linear",
                            align_corners=False,
                        )
                        # permute back to (frames, seq_len, features)
                        audio_embeds = audio_embeds_interpolated.permute(2, 0, 1)

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
                    # Use valid_latents here
                    unet_input = torch.cat([valid_latents] * 2) if do_classifier_free_guidance else valid_latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    # Use valid_latents' shape to ensure consistency if guidance is off
                    current_batch_size = valid_latents.shape[0] 
                    mask_latents_input = mask_latents[:current_batch_size]
                    masked_image_latents_input = masked_image_latents[:current_batch_size]
                    ref_latents_input = ref_latents[:current_batch_size]
                    
                    # If guidance is on, double the conditioning inputs as well
                    if do_classifier_free_guidance:
                        mask_latents_input = torch.cat([mask_latents_input] * 2)
                        masked_image_latents_input = torch.cat([masked_image_latents_input] * 2)
                        ref_latents_input = torch.cat([ref_latents_input] * 2)

                    unet_input = torch.cat(
                        [unet_input, mask_latents_input, masked_image_latents_input, ref_latents_input], dim=1
                    )

                    # predict the noise residual
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

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
        
        inference_loop_end_time = time.time()
        print(f"Main inference loop total time: {inference_loop_end_time - inference_loop_start_time:.2f} seconds")

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

        restore_video_start_time = time.time()
        # Pass potentially None concatenated_frames if no valid frames were processed across all chunks
        synced_video_frames = self.restore_video(concatenated_frames, video_frames, boxes, affine_matrices, face_detected_flags)
        restore_video_end_time = time.time()
        print(f"Video restoration time: {restore_video_end_time - restore_video_start_time:.2f} seconds")


        if is_train:
            self.unet.train()

        ffmpeg_start_time = time.time()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_video_path = os.path.join(temp_dir, "video.mp4")
        write_video(temp_video_path, synced_video_frames, fps=video_fps)

        command = f"ffmpeg -y -loglevel error -nostdin -i \"{temp_video_path}\" -i \"{audio_path}\" -c:v libx264 -preset veryfast -crf 18 -c:a copy -pix_fmt yuv420p -shortest \"{video_out_path}\""
        subprocess.run(command, shell=True)
        ffmpeg_end_time = time.time()
        print(f"Final video writing and ffmpeg merging time: {ffmpeg_end_time - ffmpeg_start_time:.2f} seconds")

        call_end_time = time.time()
        print(f"Total __call__ time: {call_end_time - call_start_time:.2f} seconds")
