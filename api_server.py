import os
import uuid
import tempfile
import argparse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from omegaconf import OmegaConf
from scripts.inference import main

app = FastAPI()

@app.post("/process/")
async def process_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    inference_steps: int = Form(20),
    guidance_scale: float = Form(1.0),
    seed: int = Form(1247)
):
    # Use the system's temporary directory (works on Windows too)
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{video.filename}")
    audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{audio.filename}")
    output_video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_output.mp4")
    
    # Save uploaded files
    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    
    # Build a minimal argparse.Namespace to pass parameters to your inference main() function
    args = argparse.Namespace(
        unet_config_path="configs/unet/stage2.yaml",  # adjust as needed
        inference_ckpt_path="checkpoints/latentsync_unet.pt",
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=output_video_path,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    
    config = OmegaConf.load(args.unet_config_path)
    
    # Call the existing inference function
    main(config, args)
    
    # Optionally remove temporary input files
    os.remove(video_path)
    os.remove(audio_path)
    
    # Return the output video
    return FileResponse(output_video_path, media_type="video/mp4", filename="output_video.mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
