import os
import uuid
import time
import argparse
import asyncio
from enum import Enum
from typing import Dict, Optional, List
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, JSONResponse
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

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class QueueManager:
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.tasks: Dict[str, Dict] = {}
        self.processing_task: Optional[asyncio.Task] = None
        self.current_task_id: Optional[str] = None
        
    async def start_worker(self):
        while True:
            task_id, task_data = await self.task_queue.get()
            self.current_task_id = task_id
            self.tasks[task_id]["status"] = TaskStatus.PROCESSING
            
            try:
                if self.tasks[task_id]["status"] != TaskStatus.CANCELED:
                    result = await self._process_task(task_data)
                    self.tasks[task_id]["status"] = TaskStatus.COMPLETED
                    self.tasks[task_id]["result"] = result
            except Exception as e:
                self.tasks[task_id]["status"] = TaskStatus.FAILED
                self.tasks[task_id]["error"] = str(e)
            finally:
                self.task_queue.task_done()
                self.current_task_id = None
    
    async def _process_task(self, task_data):
        video_path = task_data["video_path"]
        audio_path = task_data["audio_path"]
        output_video_path = task_data["output_video_path"]
        args = task_data["args"]

        # Check that the required config and checkpoint files exist
        if not os.path.exists(args.unet_config_path):
            raise Exception("Config file not found")
        if not os.path.exists(args.inference_ckpt_path):
            raise Exception("Checkpoint file not found")

        config = OmegaConf.load(args.unet_config_path)
        
        # Run the inference in a separate thread to not block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: main(config, args))

        if not os.path.exists(output_video_path):
            raise Exception("Output file was not created")
            
        return output_video_path
        
    async def add_task(self, task_data):
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "added_at": datetime.now(),
            "data": task_data
        }
        await self.task_queue.put((task_id, task_data))
        return task_id
        
    def cancel_task(self, task_id):
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
            
        if self.tasks[task_id]["status"] == TaskStatus.PENDING:
            self.tasks[task_id]["status"] = TaskStatus.CANCELED
            return True
        elif self.tasks[task_id]["status"] == TaskStatus.PROCESSING and task_id == self.current_task_id:
            # For currently processing task, we can only mark it
            # but actual cancellation depends on the implementation
            self.tasks[task_id]["status"] = TaskStatus.CANCELED
            return True
        else:
            return False
            
    def get_queue_status(self):
        pending_tasks = []
        current_task = None
        completed_tasks = []
        
        for task_id, task_info in self.tasks.items():
            task_status = {
                "id": task_id,
                "status": task_info["status"],
                "added_at": task_info["added_at"].isoformat()
            }
            
            if task_info["status"] == TaskStatus.PENDING:
                pending_tasks.append(task_status)
            elif task_info["status"] == TaskStatus.PROCESSING:
                current_task = task_status
            elif task_info["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED]:
                completed_tasks.append(task_status)
                
        return {
            "queue_length": self.task_queue.qsize(),
            "current_task": current_task,
            "pending_tasks": pending_tasks,
            "completed_tasks": completed_tasks[-10:]  # Only return the 10 most recent
        }
    
    def get_task_status(self, task_id):
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return self.tasks[task_id]

# Initialize queue manager
queue_manager = QueueManager()

@app.on_event("startup")
async def startup_event():
    # Start the background worker
    asyncio.create_task(queue_manager.start_worker())

@app.post("/process/")
async def process_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    inference_steps: int = Form(20),
    guidance_scale: float = Form(1.0),
    seed: int = Form(1247)
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

        # Create task data
        task_data = {
            "video_path": video_path,
            "audio_path": audio_path,
            "output_video_path": output_video_path,
            "args": args
        }
        
        # Add to processing queue
        task_id = await queue_manager.add_task(task_data)
        
        return {"task_id": task_id, "message": "Task added to queue"}

    except Exception as e:
        # Cleanup any files created if something goes wrong
        for path in [video_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    try:
        task_info = queue_manager.get_task_status(task_id)
        
        # If task is completed, prepare for file download
        if task_info["status"] == TaskStatus.COMPLETED:
            return {
                "status": task_info["status"],
                "download_url": f"/download/{task_id}"
            }
        
        return {"status": task_info["status"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    try:
        result = queue_manager.cancel_task(task_id)
        if result:
            return {"message": "Task canceled successfully"}
        else:
            return {"message": "Task could not be canceled, it may have already completed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue")
async def get_queue_status():
    return queue_manager.get_queue_status()

@app.get("/download/{task_id}")
async def download_result(task_id: str, background_tasks: BackgroundTasks):
    task_info = queue_manager.get_task_status(task_id)
    
    if task_info["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    output_video_path = task_info["result"]
    
    if not os.path.exists(output_video_path):
        raise HTTPException(status_code=404, detail="Output file not found")
        
    # Define a generator to stream the output file
    def iterfile():
        with open(output_video_path, "rb") as f:
            yield from f

    # Schedule cleanup of the output file after streaming
    def cleanup():
        time.sleep(5)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
            
        # Also clean up the input files if they still exist
        video_path = task_info["data"]["video_path"]
        audio_path = task_info["data"]["audio_path"]
        for path in [video_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)
                
    background_tasks.add_task(cleanup)

    filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(iterfile(), media_type="video/mp4", headers=headers)

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
