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

import argparse
import os
from pathlib import Path
import sys

# Add project root to sys.path to allow for latentsync module import
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent # Assumes tools/ is one level down from project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from latentsync.utils.util import gather_video_paths_recursively


class FileslistWriter:
    def __init__(self, fileslist_path: str):
        self.fileslist_path = fileslist_path
        Path(self.fileslist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(fileslist_path, "w") as _:
            pass

    def append_dataset(self, dataset_dir: str):
        print(f"Scanning dataset dir: {dataset_dir}")
        if not Path(dataset_dir).exists():
            print(f"Warning: Dataset directory {dataset_dir} does not exist. Skipping.")
            return
        video_paths = gather_video_paths_recursively(dataset_dir)
        if not video_paths:
            print(f"No videos found in {dataset_dir}. Skipping append.")
            return
        with open(self.fileslist_path, "a") as f:
            for video_path in tqdm(video_paths, desc=f"Writing paths from {Path(dataset_dir).name}"):
                f.write(f"{video_path}\n")
        print(f"Finished appending paths from {dataset_dir} to {self.fileslist_path}")


if __name__ == "__main__":
    repo_root = PROJECT_ROOT # Use the globally defined PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Write a list of video files from specified base dataset directories, creating separate train and validation lists.")
    parser.add_argument(
        "--base_data_dirs", 
        type=str, 
        nargs='*', # 0 or more arguments
        default=[str(repo_root)], # Default to the project root if no argument is provided
        help=f"List of base dataset directories. Defaults to the repository root ({repo_root}) if not specified. The script searches for 'high_visual_quality/train' and 'high_visual_quality/validation' subfolders."
    )

    args = parser.parse_args()

    output_dir_name = "file_list"
    output_dir = repo_root / output_dir_name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Repository root: {repo_root}")
        print(f"Output directory for file lists: {output_dir}")
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        exit(1)

    train_fileslist_path = output_dir / "train_files.txt"
    validation_fileslist_path = output_dir / "validation_files.txt"

    # Clear files at the start or ensure FileslistWriter handles it
    # FileslistWriter already creates/clears the file in __init__
    train_writer = FileslistWriter(str(train_fileslist_path))
    validation_writer = FileslistWriter(str(validation_fileslist_path))

    # Inform user about the source of base_data_dirs
    if args.base_data_dirs == [str(repo_root)]:
        print(f"Using repository root as the base data directory (default): {repo_root}")
    else:
        print(f"Using provided base data directories: {args.base_data_dirs}")

    for base_dir_str in args.base_data_dirs:
        base_dir = Path(base_dir_str)
        
        train_data_subdir = base_dir / "high_visual_quality" / "train"
        validation_data_subdir = base_dir / "high_visual_quality" / "validation"

        print(f"Processing base directory: {base_dir}")

        if train_data_subdir.exists() and train_data_subdir.is_dir():
            train_writer.append_dataset(str(train_data_subdir))
        else:
            print(f"Train directory {train_data_subdir} not found or is not a directory. Skipping for base dir {base_dir}.")
            
        if validation_data_subdir.exists() and validation_data_subdir.is_dir():
            validation_writer.append_dataset(str(validation_data_subdir))
        else:
            print(f"Validation directory {validation_data_subdir} not found or is not a directory. Skipping for base dir {base_dir}.")

    print(f"Training file list saved to: {train_fileslist_path}")
    print(f"Validation file list saved to: {validation_fileslist_path}")
    print("Script finished.")
