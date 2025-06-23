import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import os
import base64
import sys
import cv2
import shutil

# --- RIFE Integration ---
# Add RIFE submodule to sys.path so we can import the interpolation function
script_dir = Path(__file__).parent.resolve()
rife_dir = script_dir / "ECCV2022-RIFE"
if rife_dir.exists() and str(rife_dir) not in sys.path:
    sys.path.append(str(rife_dir))

# Import the main function from our new script
try:
    from run_interpolation import main_interpolate as run_video_interpolation
except ImportError as e:
    print(f"Could not import video interpolation script: {e}")
    # Define a dummy function to prevent the app from crashing if RIFE is not set up
    def run_video_interpolation(*args, **kwargs):
        raise RuntimeError("RIFE submodule not found or not set up correctly. Please check the 'ECCV2022-RIFE' directory.")

# --- End RIFE Integration ---

# Get the directory of the current script to build absolute paths
script_dir = Path(__file__).parent.resolve()

# CONFIG_PATH is now determined dynamically based on user selection in the UI.


# Helper function to encode image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Prepare the logo
try:
    logo_path = script_dir / "assets/goAVA_Logo.png"
    logo_base64 = image_to_base64(logo_path)
    logo_data_uri = f"data:image/png;base64,{logo_base64}"
except FileNotFoundError:
    logo_data_uri = "" # Set to empty string if logo not found
    print(f"Warning: Logo file not found at {logo_path}. The logo will not be displayed.")


# Helper function to get checkpoint files
def get_checkpoint_files():
    # Use script_dir to build absolute paths for scanning
    base_dirs_to_scan = [
        script_dir / "checkpoints",
        script_dir / "debug",
    ]
    collected_paths_str = set()

    for p_dir in base_dirs_to_scan:
        if p_dir.exists() and p_dir.is_dir(): # Ensure directory exists
            for f_path in p_dir.rglob("*.pt"):
                # Store path as a string, relative to the script directory
                collected_paths_str.add(f_path.relative_to(script_dir).as_posix())
    
    # Sort for consistent order in the dropdown
    sorted_paths = sorted(list(collected_paths_str))
    return sorted_paths


def get_result_sessions():
    """Scans for and returns a list of past result session directories."""
    results_dir = script_dir / "results"
    if not results_dir.exists():
        return []
    # Return sorted list of session directories, newest first
    return sorted([d.name for d in results_dir.iterdir() if d.is_dir()], reverse=True)


def get_session_video(session_name):
    """Given a session name, finds the final video."""
    if not session_name:
        return gr.update(value=None, visible=False)

    session_dir = script_dir / "results" / session_name
    final_video_path = session_dir / "final_video.mp4"
    
    video_path_str = str(final_video_path) if final_video_path.exists() else None

    # Return update to show the video player and load the video if it exists
    return gr.update(value=video_path_str, visible=video_path_str is not None)


def create_thumbnail(video_path, thumbnail_path):
    """Creates a thumbnail from the first frame of a video."""
    if Path(thumbnail_path).exists():
        return True
    try:
        vidcap = cv2.VideoCapture(str(video_path))
        success, image = vidcap.read()
        if success:
            cv2.imwrite(str(thumbnail_path), image)
            vidcap.release()
            return True
        vidcap.release()
        print(f"Failed to read frame from {video_path} for thumbnail.")
        return False
    except Exception as e:
        print(f"Error creating thumbnail for {video_path}: {e}")
        return False


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
    enable_upscale,
    sharpness_factor,
    selected_checkpoint, # Added new argument, will be relative path string or "No checkpoints available"
    resolution,
    enable_interpolation,
):
    if selected_checkpoint == "No checkpoints available": # Check for placeholder string
        raise gr.Error("No checkpoint selected. Please ensure checkpoint files are available and one is selected.")

    # --- 1. Setup paths ---
    results_dir = script_dir / "results"
    temp_dir = script_dir / "temp"
    results_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)

    # Unique session folder for the final result
    video_file_path = Path(video_path)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = results_dir / f"{video_file_path.stem}_{current_time}"
    session_dir.mkdir(exist_ok=True)

    # The pre-interpolated output will be in the temp dir with a unique name
    temp_video_path = temp_dir / f"{video_file_path.stem}_{current_time}.mp4"

    # Convert source paths to absolute Path objects and normalize them
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()
    
    # Reconstruct the full, absolute checkpoint path from the selected relative path string
    checkpoint_file_obj = script_dir / selected_checkpoint
    if not checkpoint_file_obj.exists() or not checkpoint_file_obj.is_file():
        raise gr.Error(f"Selected checkpoint file not found or is not a file: {checkpoint_file_obj}")


    # Determine config path based on resolution
    if resolution == 256:
        config_path = script_dir / "configs/unet/stage2.yaml"
    elif resolution == 512:
        config_path = script_dir / "configs/unet/stage2_512.yaml"
    else:
        raise gr.Error(f"Unsupported resolution: {resolution}. Please select 256 or 512.")

    if not config_path.exists():
        raise gr.Error(f"Configuration file for resolution {resolution} not found at {config_path}")


    config = OmegaConf.load(config_path)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "enable_upscale": enable_upscale,
            "sharpness_factor": sharpness_factor,
        }
    )

    # --- 2. Parse arguments and run main processing ---
    args = create_args(
        video_path,
        audio_path,
        str(temp_video_path), # Output path for the initial run is in temp
        inference_steps,
        guidance_scale,
        seed,
        enable_upscale,
        sharpness_factor,
        checkpoint_file_obj.as_posix(),
        config_path.as_posix(),
        str(temp_dir), # Use the general temp directory for intermediate files
    )

    try:
        main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        
        # --- 3. Handle final output and interpolation ---
        final_video_path_in_session = session_dir / "final_video.mp4"

        if enable_interpolation:
            print("Interpolating video to 50 FPS...")
            try:
                # The interpolation script saves its output in the specified directory.
                interpolated_path_str = run_video_interpolation(
                    input_video_path=str(temp_video_path),
                    output_dir_path=str(session_dir)
                )
                interpolated_path_obj = Path(interpolated_path_str)
                # Rename the generated file to our consistent name
                interpolated_path_obj.rename(final_video_path_in_session)
                
                # Clean up the pre-interpolated video from the temp folder
                temp_video_path.unlink(missing_ok=True)

            except Exception as e:
                print(f"Error during video frame interpolation: {e}")
                # Clean up temp file on failure
                temp_video_path.unlink(missing_ok=True)
                raise gr.Error(f"Video interpolation failed: {e}. No video was saved.")
        else:
            # If no interpolation, move the generated video from temp to its session folder
            shutil.move(str(temp_video_path), str(final_video_path_in_session))

        # --- 4. Create thumbnail and return final path ---
        thumbnail_path = session_dir / "thumbnail.jpg"
        create_thumbnail(str(final_video_path_in_session), thumbnail_path)

        return str(final_video_path_in_session)

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        # Clean up temp file on failure
        temp_video_path.unlink(missing_ok=True)
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
    video_path: str,
    audio_path: str,
    output_path: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    enable_upscale: bool,
    sharpness_factor: float,
    checkpoint_path: str, # Added new argument
    unet_config_path: str,
    temp_dir_path: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_upscale", action='store_true', help="Enable upscale")
    parser.add_argument("--sharpness_factor", type=float, default=1.5)
    parser.add_argument("--enable_deepcache", action="store_true")

    # Build the argument list dynamically
    args_list = [
        "--unet_config_path",
        unet_config_path,
        "--inference_ckpt_path",
        checkpoint_path, # Use the passed checkpoint_path
        "--video_path",
        video_path,
        "--audio_path",
        audio_path,
        "--video_out_path",
        output_path,
        "--inference_steps",
        str(inference_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--seed",
        str(seed),
        "--sharpness_factor",
        str(sharpness_factor),
        "--temp_dir",
        temp_dir_path,
        "--enable_deepcache",
    ]
    # Add the flag only if enable_upscale is True
    if enable_upscale:
        args_list.append("--enable_upscale")

    return parser.parse_args(args_list)


# --- Gradio UI ---

def clear_all():
    """Clears all input and output fields."""
    return None, None, None

# CSS for dark theme, inspired by LivePortrait
dark_theme_css = """
:root {
    --primary-50: #eff6ff;
    --primary-100: #dbeafe;
    --primary-200: #bfdbfe;
    --primary-300: #93c5fd;
    --primary-400: #60a5fa;
    --primary-500: #3b82f6;
    --primary-600: #2563eb;
    --primary-700: #1d4ed8;
    --primary-800: #1e40af;
    --primary-900: #1e3a8a;
    --primary-950: #172554;
    --secondary-500: #8b5cf6;
}
body {
    background-color: #0B0F19;
    color: white;
}
.gradio-container {
    background-color: #0B0F19;
}
.gradio-group, .gradio-accordion {
    border: 1px solid #374151; /* neutral-700 */
    background-color: #1F2937; /* neutral-800 */
    border-radius: 8px;
}
.gradio-button {
    background-color: #4F46E5; /* A shade of purple/blue */
    color: white;
    border: none;
}
.gradio-button:hover {
    background-color: #6366F1;
}
"""

# Create Gradio interface
with gr.Blocks(css=dark_theme_css, title="go AVA Dubbing Tool") as demo:
    with gr.Tabs():
        with gr.TabItem("Dubbing Tool"):
            gr.Markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
                <img src="{logo_data_uri}" alt="goAVA Logo" style="width: 200px; height: auto; margin-bottom: 20px;">
                <h1>go AVA Dubbing Tool</h1>
                <p>Synchronize lip movements in a video with a new audio track.</p>
            </div>
            """
            )
            
            with gr.Accordion("Notices and Best Practices", open=True):
                gr.Markdown(
                """
                - **Audio/Video Matching:** For best results, select audio and video of similar length. The tool syncs mouth movements to the new audio by matching speech and silence, so a close match is ideal.
                - **There is currently a limit of 40s of 4k video as longer videos will cause the tool to crash.**
                - **(Optional) Prepare your video with 25 FPS and audio with 16000 Hz to speed up the processing time.**
                - **Make sure the file name does not contain any special characters or spaces.**
                - **You should consider to interpolate the output video to 50 FPS or more to make it smoother. (TODO: implement this in the future)**
                - **If you encounter any issues, please contact jonathan@goava.ai**
                """
                )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Upload Source Video")
                    video_input = gr.Video(label="Source Video")
                with gr.Column():
                    gr.Markdown("### Step 2: Upload Target Audio")
                    audio_input = gr.Audio(label="Target Audio", type="filepath")

            # Add checkpoint selection dropdown logic
            checkpoint_files_list = get_checkpoint_files()
            if not checkpoint_files_list:
                gr.Warning("No checkpoint files (.pt) found. Please add checkpoint files.")
                dropdown_choices = ["No checkpoints available"]
                dropdown_default_value = dropdown_choices[0]
            else:
                dropdown_choices = checkpoint_files_list
                preferred_default = "checkpoints/default_unet_v1.5.pt"
                if preferred_default in dropdown_choices:
                    dropdown_default_value = preferred_default
                else:
                    dropdown_default_value = dropdown_choices[0]

            with gr.Accordion("Advanced Options & Checkpoint Selection", open=True):
                with gr.Group():
                    gr.Markdown(
                        """
                        - **Checkpoint Selection:** Use character-specific checkpoints if available (e.g., `debug/unet/character_name/checkpoint-10000.pt`). The number indicates training steps; higher usually means better, but feel free to experiment. 1.5 version checkpoints are for 256 resolution and 1.6 version checkpoints are for 512 resolution. The 1.6 checkpoints provide better quality but are slower to process.
                        """
                        )
                    checkpoint_dropdown = gr.Dropdown(
                        choices=dropdown_choices,
                        value=dropdown_default_value,
                        label="UNet Checkpoint",
                    )
                with gr.Group():
                    gr.Markdown("Select generation resolution. **Ensure your selected checkpoint matches the resolution.**")
                    resolution_input = gr.Radio(
                        [256, 512],
                        value=256,
                        label="Resolution",
                        info="For 1.5 version checkpoints, use 256. For 1.6 version checkpoints, use 512."
                    )
                with gr.Group():
                    gr.Markdown("Adjust generation parameters.")
                    with gr.Row():
                         guidance_scale = gr.Slider(minimum=1.0, maximum=2.5, value=2.5, step=0.1, label="Guidance Scale")
                         inference_steps = gr.Slider(minimum=10, maximum=50, value=50, step=1, label="Inference Steps")
                    gr.Markdown(
                        """
                        - **Guidance Scale:** Controls how strictly the lip movements follow the audio.
                        - **Inference Steps:** More steps can improve quality but increase processing time.
                        """
                    )
                    with gr.Row():
                        enable_upscale = gr.Checkbox(value=True, label="Enable Upscale")
                        sharpness_factor = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Sharpness Factor")
                    
                    enable_interpolation = gr.Checkbox(value=True, label="Enable 50 FPS Video Frame Interpolation")
                    gr.Markdown(
                        """
                        - **Enable 50 FPS Interpolation:** Doubles the frame rate of the output video from 25 to 50 FPS using video frame interpolation. This can make motion appear smoother but will increase processing time.
                        """
                    )

                    seed = gr.Number(value=1247, label="Random Seed", precision=0)
                    gr.Markdown(
                        """
                        - **Random Seed:** A fixed seed ensures reproducible results for the same inputs.
                        """
                    )

            with gr.Row():
                clear_btn = gr.Button("Clear")
                process_btn = gr.Button("Process Video", variant="primary")

            with gr.Column():
                gr.Markdown("### Step 3: View Dubbed Video")
                video_output = gr.Video(label="Output Video")


            # --- Button Clicks ---
            process_btn.click(
                fn=process_video,
                inputs=[
                    video_input,
                    audio_input,
                    guidance_scale,
                    inference_steps,
                    seed,
                    enable_upscale,
                    sharpness_factor,
                    checkpoint_dropdown,
                    resolution_input,
                    enable_interpolation,
                ],
                outputs=video_output,
            )
            
            clear_btn.click(fn=clear_all, inputs=None, outputs=[video_input, audio_input, video_output])

        with gr.TabItem("Results Browser"):
            gr.Markdown("## Browse and Download Past Results")
            gr.Markdown("Select a result from the gallery to preview the generated videos. The list is sorted with the most recent results first.")
            
            # Invisible textbox to store the session name selected from the gallery
            selected_session_name = gr.Textbox(visible=False)

            with gr.Row():
                gallery = gr.Gallery(
                    label="Past Results",
                    show_label=False,
                    elem_id="gallery",
                    columns=[5], 
                    object_fit="contain",
                    height="auto"
                )
            
            with gr.Row():
                refresh_button = gr.Button("Refresh Results")

            with gr.Row():
                video_output_browser = gr.Video(label="Result Video", visible=False)

            def get_gallery_data():
                """Scans the results directory and prepares data for the gallery."""
                results_dir = script_dir / "results"
                if not results_dir.exists():
                    return []
                
                sessions = sorted(
                    [d for d in results_dir.iterdir() if d.is_dir()], 
                    reverse=True
                )
                
                gallery_items = []
                for session in sessions:
                    thumbnail = session / "thumbnail.jpg"
                    final_video = session / "final_video.mp4"

                    # If thumbnail doesn't exist, try to create it
                    if not thumbnail.exists() and final_video.exists():
                        create_thumbnail(final_video, thumbnail)

                    if thumbnail.exists():
                        gallery_items.append((str(thumbnail), session.name))
                return gallery_items

            def on_gallery_select(gallery_data: list, evt: gr.SelectData):
                """Handles the selection of an item in the gallery."""
                # `evt.index` gives the index of the selected item
                # `gallery_data` is the list of (filepath, caption) tuples
                selected_item_caption = gallery_data[evt.index][1]
                return selected_item_caption

            def refresh_gallery():
                """Refreshes the gallery with the latest results."""
                return gr.update(value=get_gallery_data())

            # --- Event Handlers for Results Browser ---
            demo.load(get_gallery_data, None, gallery)

            refresh_button.click(
                fn=refresh_gallery,
                inputs=None,
                outputs=[gallery]
            )

            gallery.select(
                fn=on_gallery_select,
                inputs=[gallery],
                outputs=[selected_session_name],
                show_progress="hidden"
            )

            selected_session_name.change(
                fn=get_session_video,
                inputs=[selected_session_name],
                outputs=[video_output_browser]
            )

if __name__ == "__main__":
    # --- Authentication ---
    # Define username and password directly in the script
    # !! Warning: Storing credentials directly in code is insecure !!
    # !! Change these values directly here if needed              !!
    AUTH_USERNAME = "admin"
    AUTH_PASSWORD = "goAVA_2025"

    auth_creds = (AUTH_USERNAME, AUTH_PASSWORD)
    print(f"Authentication enabled for user: {AUTH_USERNAME}")
    # --- End Authentication ---

    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        inbrowser=False,
        auth=auth_creds, # Pass credentials tuple to enable authentication
    )
