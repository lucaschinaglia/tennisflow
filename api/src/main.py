import os
import json
import uuid
import logging
import tempfile
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import supabase
from . import processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="TennisFlow Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=0
)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

# Models
class VideoInfo(BaseModel):
    video_id: str
    video_url: str
    user_id: str
    
class AnalysisResponse(BaseModel):
    task_id: str
    status: str
    message: str
    
class AnalysisStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    result_url: Optional[str] = None

# Routes
@app.get("/")
async def root():
    return {"message": "TennisFlow Analysis API is running"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(video_info: VideoInfo, background_tasks: BackgroundTasks):
    """
    Queue a video for tennis swing analysis
    """
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Store task in Redis
        task_data = {
            "task_id": task_id,
            "video_id": video_info.video_id,
            "video_url": video_info.video_url,
            "user_id": video_info.user_id,
            "status": "queued",
            "progress": 0,
            "created_at": processor.get_timestamp()
        }
        
        # Update video status in Supabase
        supabase_client.table('videos').update({
            "analysis_status": "processing"
        }).eq('id', video_info.video_id).execute()
        
        # Store task data in Redis
        redis_client.set(f"task:{task_id}", json.dumps(task_data))
        
        # Add to processing queue
        redis_client.lpush("processing_queue", task_id)
        
        logger.info(f"Video {video_info.video_id} queued for analysis with task ID {task_id}")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Video analysis has been queued"
        }
    except Exception as e:
        logger.error(f"Error queueing video analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(task_id: str):
    """
    Get the status of a video analysis task
    """
    try:
        # Get task data from Redis
        task_data_str = redis_client.get(f"task:{task_id}")
        
        if not task_data_str:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = json.loads(task_data_str)
        
        return {
            "task_id": task_id,
            "status": task_data.get("status", "unknown"),
            "progress": task_data.get("progress", 0),
            "message": task_data.get("message", ""),
            "result_url": task_data.get("result_url")
        }
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_id}/analysis")
async def get_video_analysis(video_id: str):
    """
    Get the analysis results for a specific video
    """
    try:
        # Get the video details from Supabase
        response = supabase_client.table('videos').select('*').eq('id', video_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video = response.data[0]
        
        if video["analysis_status"] != "completed":
            return {
                "status": video["analysis_status"],
                "message": f"Analysis is {video['analysis_status']}"
            }
        
        # Return the analysis results
        return {
            "status": "completed",
            "results": video["analysis_results"]
        }
    except Exception as e:
        logger.error(f"Error getting video analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    title: str = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a video and queue it for analysis
    """
    try:
        # Generate video ID
        video_id = str(uuid.uuid4())
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(await file.read())
        temp_file.close()
        
        # Upload to Supabase storage
        video_path = f"{user_id}/{video_id}/video.mp4"
        with open(temp_file.name, "rb") as f:
            supabase_client.storage.from_("videos").upload(
                video_path,
                f,
                file_options={"content-type": "video/mp4"}
            )
        
        # Get public URL
        video_url = supabase_client.storage.from_("videos").get_public_url(video_path)
        
        # Create video record in database
        video_data = {
            "id": video_id,
            "user_id": user_id,
            "title": title,
            "description": description,
            "video_url": video_url,
            "analysis_status": "pending"
        }
        
        supabase_client.table('videos').insert(video_data).execute()
        
        # Queue for analysis
        background_tasks.add_task(
            analyze_video,
            VideoInfo(video_id=video_id, video_url=video_url, user_id=user_id),
            background_tasks
        )
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        return {
            "video_id": video_id,
            "status": "uploaded",
            "message": "Video uploaded and queued for analysis"
        }
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    # Check Redis connection
    try:
        redis_client.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    # Check Supabase connection
    try:
        supabase_client.table('videos').select('count').limit(1).execute()
        supabase_status = "ok"
    except Exception as e:
        supabase_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "services": {
            "redis": redis_status,
            "supabase": supabase_status
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)