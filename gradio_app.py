import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import os # Added for listing files

CONFIG_PATH = Path("configs/unet/stage2.yaml")
# CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt") # Removed, will be dynamic


# Helper function to get checkpoint files
def get_checkpoint_files():
    base_dirs_to_scan = [
        Path("checkpoints"),
        Path("debug"),
    ]
    collected_paths_str = set()

    for p_dir in base_dirs_to_scan:
        if p_dir.exists() and p_dir.is_dir(): # Ensure directory exists
            if p_dir.name == "debug": # Recursive search for 'debug' directory
                for f_path in p_dir.rglob("*.pt"):
                    collected_paths_str.add(f_path.as_posix())
            else: # Non-recursive for other specified dirs like 'checkpoints'
                for f_path in p_dir.glob("*.pt"):
                    collected_paths_str.add(f_path.as_posix())
    
    # Sort for consistent order in the dropdown
    sorted_paths = sorted(list(collected_paths_str))
    return sorted_paths


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
    enable_upscale,
    sharpness_factor,
    selected_checkpoint, # Added new argument, will be relative path string or "No checkpoints available"
):
    if selected_checkpoint == "No checkpoints available": # Check for placeholder string
        raise gr.Error("No checkpoint selected. Please ensure checkpoint files are available and one is selected.")

    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()
    
    # Construct the full checkpoint path from the selected relative path string
    checkpoint_file_obj = Path(selected_checkpoint) # selected_checkpoint is now like "checkpoints/model.pt"
    if not checkpoint_file_obj.exists() or not checkpoint_file_obj.is_file():
        raise gr.Error(f"Selected checkpoint file not found or is not a file: {selected_checkpoint}")


    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4") # Change the filename as needed

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "enable_upscale": enable_upscale,
            "sharpness_factor": sharpness_factor,
        }
    )

    # Parse the arguments
    args = create_args(
        video_path,
        audio_path,
        output_path,
        inference_steps,
        guidance_scale,
        seed,
        enable_upscale,
        sharpness_factor,
        checkpoint_file_obj.absolute().as_posix(), # Pass the selected checkpoint path (now an absolute path string)
    )

    try:
        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        return output_path # Ensure the output path is returned
    except Exception as e:
        print(f"Error during processing: {str(e)}")
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
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_upscale", action='store_true', help="Enable upscale")
    parser.add_argument("--sharpness_factor", type=float, default=1.5)

    # Build the argument list dynamically
    args_list = [
        "--unet_config_path",
        CONFIG_PATH.absolute().as_posix(),
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
    gr.Markdown(
    """
    <div style="text-align: center;">
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
        preferred_default = "checkpoints/latentsync_unet.pt"
        if preferred_default in dropdown_choices:
            dropdown_default_value = preferred_default
        else:
            dropdown_default_value = dropdown_choices[0]

    with gr.Accordion("Advanced Options & Checkpoint Selection", open=False):
        with gr.Group():
            gr.Markdown(
                """
                - **Checkpoint Selection:** Use character-specific checkpoints if available (e.g., `debug/unet/character_name/checkpoint-10000.pt`). The number indicates training steps; higher usually means better, but feel free to experiment.
                """
                )
            checkpoint_dropdown = gr.Dropdown(
                choices=dropdown_choices,
                value=dropdown_default_value,
                label="UNet Checkpoint",
            )
        with gr.Group():
            gr.Markdown("Adjust generation parameters.")
            with gr.Row():
                 guidance_scale = gr.Slider(minimum=1.0, maximum=2.5, value=1.2, step=0.1, label="Guidance Scale")
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
            gr.Markdown(
                """
                - **Enable Upscale:** Upscales the output video resolution.
                - **Sharpness Factor:** Adjusts sharpness of the upscaled video.
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
        ],
        outputs=video_output,
    )
    
    clear_btn.click(fn=clear_all, inputs=None, outputs=[video_input, audio_input, video_output])


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
        auth=auth_creds # Pass credentials tuple to enable authentication
    )
