import os
import uuid
import tempfile
import argparse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from omegaconf import OmegaConf
from scripts.inference import main

app = FastAPI()

# Define an output directory (inside the system temp directory) to store generated videos.
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "latentsync_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/process/")
async def process_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    inference_steps: int = Form(20),
    guidance_scale: float = Form(1.0),
    seed: int = Form(1247)
):
    # Use the system's temporary directory for input files.
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{video.filename}")
    audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{audio.filename}")
    
    # Use the persistent output directory for the generated video.
    output_filename = f"{uuid.uuid4()}_output.mp4"
    output_video_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save the uploaded video and audio files.
    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    
    # Build an argparse.Namespace as expected by your inference main() function.
    args = argparse.Namespace(
        unet_config_path="configs/unet/stage2.yaml",  # adjust if needed
        inference_ckpt_path="checkpoints/latentsync_unet.pt",
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=output_video_path,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    
    config = OmegaConf.load(args.unet_config_path)
    
    # Call the inference function.
    main(config, args)
    
    # Clean up the temporary input files.
    os.remove(video_path)
    os.remove(audio_path)
    
    # Return a JSON response with a download URL.
    return JSONResponse({"download_url": f"/download/{output_filename}"})

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Stream the file in chunks.
    def iterfile():
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk
    
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(iterfile(), media_type="video/mp4", headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
