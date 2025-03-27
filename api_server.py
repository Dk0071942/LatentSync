import os
import uuid
import time
import argparse
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from omegaconf import OmegaConf
from scripts.inference import main
import uvicorn
from io import BytesIO

app = FastAPI()

# Directories for storing uploads and outputs
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/process/")
async def process_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    inference_steps: int = Form(20),
    guidance_scale: float = Form(1.0),
    seed: int = Form(1247),
    background_tasks: BackgroundTasks = None
):
    # Save the uploaded video and audio files
    video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{video.filename}")
    audio_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{audio.filename}")
    output_video_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_output.mp4")

    try:
        video_content = await video.read()
        with open(video_path, "wb") as f:
            f.write(video_content)
        audio_content = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_content)

        # Build arguments for the inference process
        args = argparse.Namespace(
            unet_config_path="configs/unet/stage2.yaml",
            inference_ckpt_path="checkpoints/latentsync_unet.pt",
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_video_path,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # Check that the required config and checkpoint files exist
        if not os.path.exists(args.unet_config_path):
            raise HTTPException(status_code=500, detail="Config file not found")
        if not os.path.exists(args.inference_ckpt_path):
            raise HTTPException(status_code=500, detail="Checkpoint file not found")

        config = OmegaConf.load(args.unet_config_path)
        main(config, args)

        if not os.path.exists(output_video_path):
            raise HTTPException(status_code=500, detail="Output file was not created")

        # Clean up input files
        os.remove(video_path)
        os.remove(audio_path)

        # Define a generator to stream the output file
        def iterfile():
            with open(output_video_path, "rb") as f:
                yield from f

        # Optionally schedule cleanup of the output file after streaming
        if background_tasks:
            def cleanup():
                time.sleep(5)
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)
            background_tasks.add_task(cleanup)

        filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        headers = {"Content-Disposition": f"attachment; filename={filename}"}
        return StreamingResponse(iterfile(), media_type="video/mp4", headers=headers)

    except Exception as e:
        # Cleanup any files created if something goes wrong
        for path in [video_path, audio_path, output_video_path]:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))


# Modified test endpoint that returns a dummy file
@app.get("/test-save/")
async def test_save():
    # Create a dummy file in memory (this is just sample data)
    dummy_content = b"This is a dummy MP4 file content for testing."
    dummy_stream = BytesIO(dummy_content)
    headers = {"Content-Disposition": "attachment; filename=dummy_test_video.mp4"}
    return StreamingResponse(dummy_stream, media_type="video/mp4", headers=headers)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
