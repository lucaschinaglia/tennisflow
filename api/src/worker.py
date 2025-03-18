import os
import sys
import time
import json
import logging
import tempfile
import shutil
import uuid
import urllib.request
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
import redis
import supabase
from processor import (
    download_video, 
    extract_frames, 
    analyze_frames,
    generate_analysis_results,
    get_timestamp
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# OpenPose executable path
OPENPOSE_PATH = "/opt/openpose/build/examples/openpose/openpose.bin"

class Worker:
    def __init__(self):
        self.running = True
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Worker initialized with temp directory: {self.temp_dir}")
        
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directory")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
    
    def process_task(self, task_id: str) -> bool:
        """Process a single video analysis task"""
        try:
            # Get task data from Redis
            task_data_str = redis_client.get(f"task:{task_id}")
            if not task_data_str:
                logger.error(f"Task {task_id} not found")
                return False
            
            task_data = json.loads(task_data_str)
            video_id = task_data["video_id"]
            video_url = task_data["video_url"]
            
            logger.info(f"Processing video {video_id} from {video_url}")
            
            # Update task status
            task_data["status"] = "processing"
            task_data["progress"] = 0.1
            task_data["started_at"] = get_timestamp()
            redis_client.set(f"task:{task_id}", json.dumps(task_data))
            
            # Create a unique working directory for this task
            work_dir = os.path.join(self.temp_dir, f"task_{task_id}")
            os.makedirs(work_dir, exist_ok=True)
            
            # Download video
            video_path = os.path.join(work_dir, "video.mp4")
            download_video(video_url, video_path)
            logger.info(f"Downloaded video to {video_path}")
            
            # Update progress
            task_data["progress"] = 0.2
            redis_client.set(f"task:{task_id}", json.dumps(task_data))
            
            # Extract frames
            frames_dir = os.path.join(work_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            frame_paths = extract_frames(video_path, frames_dir)
            logger.info(f"Extracted {len(frame_paths)} frames")
            
            # Update progress
            task_data["progress"] = 0.3
            redis_client.set(f"task:{task_id}", json.dumps(task_data))
            
            # Analyze frames with OpenPose or MediaPipe (in a real implementation)
            # Here, we'll either use:
            # 1. MediaPipe (simpler, easier to integrate, less accurate)
            # 2. OpenPose (more accurate, harder to integrate)
            
            use_openpose = os.path.exists(OPENPOSE_PATH)
            
            if use_openpose:
                # Run OpenPose on the frames directory
                logger.info("Using OpenPose for pose detection")
                keypoints_dir = os.path.join(work_dir, "keypoints")
                os.makedirs(keypoints_dir, exist_ok=True)
                
                # Run OpenPose command line tool
                cmd = [
                    OPENPOSE_PATH,
                    "--image_dir", frames_dir,
                    "--write_json", keypoints_dir,
                    "--display", "0",
                    "--render_pose", "0",
                    "--model_pose", "BODY_25"
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait for OpenPose to finish
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"OpenPose error: {stderr.decode('utf-8')}")
                    raise Exception("OpenPose processing failed")
                
                # Load OpenPose results
                frame_poses = []
                for i, frame_path in enumerate(sorted(frame_paths)):
                    frame_name = os.path.basename(frame_path).split('.')[0]
                    keypoint_path = os.path.join(keypoints_dir, f"{frame_name}_keypoints.json")
                    
                    if os.path.exists(keypoint_path):
                        with open(keypoint_path, 'r') as f:
                            keypoints_data = json.load(f)
                        
                        # Format OpenPose data into our expected format
                        if keypoints_data.get("people") and len(keypoints_data["people"]) > 0:
                            person_data = keypoints_data["people"][0]
                            pose_keypoints = person_data.get("pose_keypoints_2d", [])
                            
                            # OpenPose returns a flat array [x1, y1, c1, x2, y2, c2, ...]
                            # We need to convert it to our expected format
                            keypoints = []
                            for j in range(0, len(pose_keypoints), 3):
                                if j + 2 < len(pose_keypoints):
                                    x, y, confidence = pose_keypoints[j:j+3]
                                    keypoint_name = f"keypoint_{j//3}"
                                    keypoints.append({
                                        "name": keypoint_name,
                                        "position": {"x": x, "y": y},
                                        "confidence": confidence
                                    })
                            
                            frame_poses.append({
                                "frame_number": i,
                                "keypoints": keypoints
                            })
                        else:
                            logger.warning(f"No people detected in frame {i}")
                    else:
                        logger.warning(f"Keypoint file not found for frame {i}")
                
                # Update progress
                task_data["progress"] = 0.6
                redis_client.set(f"task:{task_id}", json.dumps(task_data))
            else:
                # Use MediaPipe as fallback (or for development)
                logger.info("Using MediaPipe for pose detection")
                frame_poses = analyze_frames(frame_paths)
                
                # Update progress
                task_data["progress"] = 0.6
                redis_client.set(f"task:{task_id}", json.dumps(task_data))
            
            # Generate analysis results
            analysis_results = generate_analysis_results(video_id, frame_poses)
            logger.info(f"Generated analysis results")
            
            # Update progress
            task_data["progress"] = 0.8
            redis_client.set(f"task:{task_id}", json.dumps(task_data))
            
            # Update database with results
            update_data = {
                "analysis_results": analysis_results,
                "analysis_status": "completed"
            }
            
            supabase_client.table('videos').update(update_data).eq('id', video_id).execute()
            logger.info(f"Updated analysis results in database for video {video_id}")
            
            # Insert detailed analysis data
            if "frames" in analysis_results:
                for frame in analysis_results["frames"]:
                    frame_data = {
                        "video_id": video_id,
                        "frame_number": frame["frameNumber"],
                        "timestamp": frame["timestamp"],
                        "pose_data": frame["poseData"],
                        "swing_phase": frame["swingPhase"],
                        "annotations": frame.get("annotations", [])
                    }
                    
                    supabase_client.table('analysis_details').insert(frame_data).execute()
            
            # Update task as completed
            task_data["status"] = "completed"
            task_data["progress"] = 1.0
            task_data["completed_at"] = get_timestamp()
            redis_client.set(f"task:{task_id}", json.dumps(task_data))
            
            # Clean up work directory
            shutil.rmtree(work_dir)
            
            logger.info(f"Successfully processed video {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            
            # Update task as failed
            try:
                task_data["status"] = "failed"
                task_data["message"] = str(e)
                task_data["failed_at"] = get_timestamp()
                redis_client.set(f"task:{task_id}", json.dumps(task_data))
                
                # Update video status in database
                supabase_client.table('videos').update({
                    "analysis_status": "failed"
                }).eq('id', task_data["video_id"]).execute()
            except Exception as update_error:
                logger.error(f"Error updating failed task: {update_error}")
            
            return False
    
    def run(self):
        """Main worker loop"""
        logger.info("Starting worker")
        
        while self.running:
            try:
                # Pop task from queue
                task_id = redis_client.rpop("processing_queue")
                
                if task_id:
                    task_id = task_id.decode('utf-8')
                    logger.info(f"Got task {task_id} from queue")
                    self.process_task(task_id)
                else:
                    # No tasks, sleep for a bit
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(5)  # Sleep on error to avoid tight loop
        
        self.cleanup()

if __name__ == "__main__":
    worker = Worker()
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    finally:
        worker.cleanup()