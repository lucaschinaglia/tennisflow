#!/usr/bin/env python3
"""
Run the TennisFlow API server.
"""
import os
import sys
import argparse
import logging
import uvicorn
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("tennisflow.run_api")

# Add module directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import TennisFlow modules
from models.api_integration import (
    initialize_analyzer, is_analyzer_ready, 
    analyze_video_from_path, analyze_video_from_url, 
    analyze_video_from_base64
)

# Define API models
class VideoUrl(BaseModel):
    url: HttpUrl = Field(..., max_length=2083, description="URL of the video")
    options: Optional[Dict[str, Any]] = None

class VideoBase64(BaseModel):
    data: str
    options: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    videoId: str
    duration: Optional[float] = None
    frames: List[Dict[str, Any]]
    events: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="TennisFlow API",
    description="API for tennis motion analysis using TennisFlow models",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing TennisFlow models...")
    
    # Get model paths from environment variables if available
    pose_model_path = os.environ.get(
        "TENNISFLOW_POSE_MODEL_PATH", 
        "tennisflow/models/output/pose_models/best_model.pth"
    )
    event_model_path = os.environ.get(
        "TENNISFLOW_EVENT_MODEL_PATH", 
        "tennisflow/models/output/event_models/best_model.pth"
    )
    
    # Initialize analyzer
    initialize_analyzer(
        pose_model_path=pose_model_path,
        event_model_path=event_model_path,
        device=os.environ.get("TENNISFLOW_DEVICE", "cpu")
    )

# Define routes
@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "TennisFlow API is running"}

@app.get("/status")
async def status():
    """Check if the service is ready."""
    return {
        "status": "ready" if is_analyzer_ready() else "initializing",
        "analyzer_ready": is_analyzer_ready(),
        "timestamp": time.time()
    }

@app.post("/analyze/url", response_model=AnalysisResult)
async def analyze_url(video: VideoUrl):
    """
    Analyze a video from a URL.

    Args:
        video: VideoUrl object with URL and options

    Returns:
        Analysis results
    """
    if not is_analyzer_ready():
        raise HTTPException(
            status_code=503, 
            detail="Service is initializing. Please try again later."
        )
    
    results = analyze_video_from_url(str(video.url), video.options)
    
    if "error" in results and results["error"]:
        logger.error(f"Error analyzing video: {results['error']}")
        
    return results

@app.post("/analyze/upload", response_model=AnalysisResult)
async def analyze_upload(file: UploadFile = File(...)):
    """
    Analyze an uploaded video file.

    Args:
        file: Uploaded video file

    Returns:
        Analysis results
    """
    if not is_analyzer_ready():
        raise HTTPException(
            status_code=503, 
            detail="Service is initializing. Please try again later."
        )
    
    # Save uploaded file to temporary location
    import tempfile
    import shutil
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Process the video
        results = analyze_video_from_path(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        if "error" in results and results["error"]:
            logger.error(f"Error analyzing video: {results['error']}")
            
        return results
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        logger.error(f"Error processing uploaded video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/base64", response_model=AnalysisResult)
async def analyze_base64(video: VideoBase64):
    """
    Analyze a base64 encoded video.

    Args:
        video: VideoBase64 object with data and options

    Returns:
        Analysis results
    """
    if not is_analyzer_ready():
        raise HTTPException(
            status_code=503, 
            detail="Service is initializing. Please try again later."
        )
    
    results = analyze_video_from_base64(video.data, video.options)
    
    if "error" in results and results["error"]:
        logger.error(f"Error analyzing video: {results['error']}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run TennisFlow API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting TennisFlow API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "run_api_server:app", 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
