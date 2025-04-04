#!/usr/bin/env python3
"""
Video annotation tool for TennisFlow.

This script helps annotate tennis shots in videos by marking the frame where
each shot occurs. The annotations are saved to a CSV file that can be used
for training and validation.

Based on the annotator.py from the tennis_shot_recognition repository by Antoine Keller.
"""

import os
import sys
import argparse
import csv
import cv2
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def annotate_video(video_path, output_csv=None, playback_speed=1.0):
    """
    Annotate tennis shots in a video.
    
    Args:
        video_path: Path to the input video
        output_csv: Path to save the annotation CSV file (default: auto-generated)
        playback_speed: Speed multiplier for video playback (default: 1.0)
        
    Returns:
        Path to the created annotation CSV file
    """
    # Generate default output filename if not provided
    if output_csv is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"annotation_{video_name}_{timestamp}.csv"
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate the delay between frames based on playback speed
    frame_delay = int(1000 / (fps * playback_speed))
    
    logger.info(f"Video loaded: {total_frames} frames at {fps} fps")
    logger.info(f"Video dimensions: {width}x{height}")
    logger.info(f"Playback speed: {playback_speed}x")
    logger.info(f"Output will be saved to: {output_csv}")
    print("\nControls:")
    print("  F - Annotate as Forehand")
    print("  B - Annotate as Backhand")
    print("  S - Annotate as Serve")
    print("  V - Annotate as Volley")
    print("  N - Annotate as Neutral")
    print("  Space - Pause/Resume")
    print("  Left Arrow - Go back 10 frames")
    print("  Right Arrow - Go forward 10 frames")
    print("  [ - Decrease playback speed")
    print("  ] - Increase playback speed")
    print("  Esc - Exit\n")
    
    # Initialize annotation data
    annotations = []
    frame_idx = 0
    paused = False
    
    # Create CSV file and writer
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Shot', 'FrameId', 'Timestamp'])
        
        # Main annotation loop
        while frame_idx < total_frames:
            if not paused:
                # Set position and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Error reading frame {frame_idx}")
                    break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Add information overlay
            cv2.rectangle(display_frame, (0, 0), (400, 90), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Time: {frame_idx/fps:.2f}s", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the annotated shots in this region
            annotation_window = 15  # frames
            region_shots = [a for a in annotations if abs(a[1] - frame_idx) <= annotation_window]
            if region_shots:
                cv2.putText(display_frame, "Recent annotations:", (width - 300, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_pos = 60
                for shot, shot_frame, _ in region_shots:
                    cv2.putText(display_frame, f"{shot} at frame {shot_frame}", (width - 300, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_pos += 30
            
            # Show the frame
            cv2.imshow('Video Annotation', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(0 if paused else frame_delay) & 0xFF
            
            if key == 27:  # Esc
                break
            elif key == 32:  # Space
                paused = not paused
                logger.info("Playback " + ("paused" if paused else "resumed"))
            elif key == ord('f'):
                annotations.append(('forehand', frame_idx, frame_idx/fps))
                writer.writerow(['forehand', frame_idx, frame_idx/fps])
                csvfile.flush()
                logger.info(f"Annotated forehand at frame {frame_idx}, time {frame_idx/fps:.2f}s")
            elif key == ord('b'):
                annotations.append(('backhand', frame_idx, frame_idx/fps))
                writer.writerow(['backhand', frame_idx, frame_idx/fps])
                csvfile.flush()
                logger.info(f"Annotated backhand at frame {frame_idx}, time {frame_idx/fps:.2f}s")
            elif key == ord('s'):
                annotations.append(('serve', frame_idx, frame_idx/fps))
                writer.writerow(['serve', frame_idx, frame_idx/fps])
                csvfile.flush()
                logger.info(f"Annotated serve at frame {frame_idx}, time {frame_idx/fps:.2f}s")
            elif key == ord('v'):
                annotations.append(('volley', frame_idx, frame_idx/fps))
                writer.writerow(['volley', frame_idx, frame_idx/fps])
                csvfile.flush()
                logger.info(f"Annotated volley at frame {frame_idx}, time {frame_idx/fps:.2f}s")
            elif key == ord('n'):
                annotations.append(('neutral', frame_idx, frame_idx/fps))
                writer.writerow(['neutral', frame_idx, frame_idx/fps])
                csvfile.flush()
                logger.info(f"Annotated neutral at frame {frame_idx}, time {frame_idx/fps:.2f}s")
            elif key == 81:  # Left arrow
                frame_idx = max(0, frame_idx - 10)
            elif key == 83:  # Right arrow
                frame_idx = min(total_frames - 1, frame_idx + 10)
            elif key == ord('['):
                playback_speed = max(0.1, playback_speed - 0.1)
                frame_delay = int(1000 / (fps * playback_speed))
                logger.info(f"Playback speed: {playback_speed:.1f}x")
            elif key == ord(']'):
                playback_speed = min(5.0, playback_speed + 0.1)
                frame_delay = int(1000 / (fps * playback_speed))
                logger.info(f"Playback speed: {playback_speed:.1f}x")
            
            # Advance to next frame if not paused
            if not paused:
                frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Annotation completed. Saved to {output_csv}")
    logger.info(f"Total annotations: {len(annotations)}")
    
    # Print statistics on annotated shots
    shot_counts = {}
    for shot, _, _ in annotations:
        shot_counts[shot] = shot_counts.get(shot, 0) + 1
    
    logger.info("Shot distribution:")
    for shot, count in shot_counts.items():
        logger.info(f"  {shot}: {count}")
    
    return output_csv

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Annotate tennis shots in a video")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", "-o", help="Path to save the annotation CSV file")
    parser.add_argument("--speed", "-s", type=float, default=1.0, 
                        help="Playback speed multiplier (default: 1.0)")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.isfile(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Run annotation tool
    result = annotate_video(args.video_path, args.output, args.speed)
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 