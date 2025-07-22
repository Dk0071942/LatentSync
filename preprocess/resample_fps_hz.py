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
import subprocess
import tqdm
from multiprocessing import Pool
import cv2

paths = []


def gather_paths(input_dir, output_dir):
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, video_output])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def get_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    return fps


def resample_fps_hz(video_input, video_output):
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from latentsync.utils.ffmpeg_config import get_standard_video_params, get_standard_audio_params
    
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    if get_video_fps(video_input) == 25:
        # Video already at 25fps, just copy video and re-encode audio with standard params
        audio_params = get_standard_audio_params()
        command_parts = ["ffmpeg", "-loglevel", "error", "-y", "-i", video_input, "-c:v", "copy"] + audio_params + [video_output]
        command = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command_parts)
    else:
        # Resample to 25fps and apply standard encoding
        video_params = get_standard_video_params()
        audio_params = get_standard_audio_params()
        command_parts = ["ffmpeg", "-loglevel", "error", "-y", "-i", video_input, "-r", "25"] + video_params + audio_params + [video_output]
        command = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command_parts)
    subprocess.run(command, shell=True)


def multi_run_wrapper(args):
    return resample_fps_hz(*args)


def resample_fps_hz_multiprocessing(input_dir, output_dir, num_workers):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_paths(input_dir, output_dir)

    print(f"Resampling FPS and Hz of {input_dir} ...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, paths), total=len(paths)):
            pass


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/raw"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/resampled"
    num_workers = 20

    resample_fps_hz_multiprocessing(input_dir, output_dir, num_workers)
