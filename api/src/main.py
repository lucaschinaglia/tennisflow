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
import processor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("../.env")  # Adjust path if needed

# Debug environment variables
print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
print(f"SUPABASE_KEY: {os.getenv('SUPABASE_KEY')}")
print(f"REDIS_HOST: {os.getenv('REDIS_HOST')}")

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
redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
try:
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=0
    )
    redis_available = True
    logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Will operate without Redis.")
    redis_client = None
    redis_available = False

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
# Fix: Use SUPABASE_KEY instead of SUPABASE_SERVICE_ROLE_KEY
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    logger.error(f"Missing Supabase credentials. URL: {supabase_url}, Key exists: {bool(supabase_key)}")
    
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
        
        # Check if Redis is available
        if not redis_available or not redis_client:
            logger.warning("Redis not available, cannot queue video for analysis")
            return {
                "task_id": f"no-redis-{video_info.video_id}",
                "status": "pending",
                "message": "Redis not available, but video registered for manual analysis"
            }
        
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
        try:
            supabase_client.table('videos').update({
                "analysis_status": "processing"
            }).eq('id', video_info.video_id).execute()
            logger.info(f"Updated video status to 'processing' in Supabase")
        except Exception as e:
            logger.error(f"Failed to update video status in Supabase: {e}")
        
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
        if not redis_available or not redis_client:
            # If Redis is not available, return a placeholder status
            if task_id.startswith("no-redis-"):
                # Extract video ID from the task ID
                video_id = task_id[9:]  # Remove "no-redis-" prefix
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "progress": 0,
                    "message": "Redis not available, video uploaded but not processed",
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "unknown",
                    "message": "Redis not available for status tracking"
                }
            
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
        logger.info(f"Receiving upload request for user {user_id}")
        # Generate video ID
        video_id = str(uuid.uuid4())
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(await file.read())
        temp_file.close()
        
        # Upload to Supabase storage - use tennis-videos bucket
        video_path = f"{user_id}/{video_id}/video.mp4"
        logger.info(f"Uploading to storage path: {video_path}")
        
        try:
            with open(temp_file.name, "rb") as f:
                supabase_client.storage.from_("tennis-videos").upload(
                    video_path,
                    f,
                    file_options={"content-type": "video/mp4"}
                )
            logger.info("Upload to Supabase storage successful")
        except Exception as storage_error:
            logger.error(f"Supabase storage upload error: {storage_error}")
            raise HTTPException(status_code=500, detail=f"Storage upload error: {str(storage_error)}")
        
        # Get public URL
        video_url = supabase_client.storage.from_("tennis-videos").get_public_url(video_path)
        
        # Create video record in database
        video_data = {
            "id": video_id,
            "user_id": user_id,
            "title": title,
            "description": description,
            "video_url": video_url,
            "analysis_status": "pending"
        }
        
        try:
            supabase_client.table('videos').insert(video_data).execute()
            logger.info(f"Video record created in database with ID {video_id}")
        except Exception as db_error:
            logger.error(f"Database insert error: {db_error}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        
        # Queue for analysis if Redis is available
        if redis_available:
            try:
                # Store task in Redis
                task_id = str(uuid.uuid4())
                task_data = {
                    "task_id": task_id,
                    "video_id": video_id,
                    "video_url": video_url,
                    "user_id": user_id,
                    "status": "queued",
                    "progress": 0,
                    "created_at": processor.get_timestamp()
                }
                
                # Store task data in Redis
                redis_client.set(f"task:{task_id}", json.dumps(task_data))
                
                # Add to processing queue
                redis_client.lpush("processing_queue", task_id)
                logger.info(f"Video {video_id} queued for analysis with task ID {task_id}")
            except Exception as redis_error:
                logger.warning(f"Redis error when queueing task: {redis_error}. Proceeding without Redis.")
        else:
            # If Redis is not available, still return success but note that async processing won't happen
            logger.warning("Redis not available, video uploaded but won't be processed automatically")
            task_id = f"no-redis-{video_id}"
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        return {
            "video_id": video_id,
            "task_id": task_id,
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
        if redis_available and redis_client:
            redis_client.ping()
            redis_status = "ok"
        else:
            redis_status = "unavailable"
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