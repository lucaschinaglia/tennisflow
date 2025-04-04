#!/usr/bin/env python3
"""
TennisFlow - Tennis Swing Analysis Pipeline

This is the main entry point for the TennisFlow application that analyzes tennis 
swing videos using pose estimation, temporal smoothing, RNN classification, and 
kinematic analysis.
"""

import os
import argparse
import logging
import json
import sys
from typing import Dict, List, Optional, Any

from src.pipeline.coordinator import PipelineCoordinator
from src.scripts.prepare_data import DataPreparation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tennisflow.log')
    ]
)
logger = logging.getLogger('tennisflow')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TennisFlow - Tennis Swing Analysis Pipeline')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Parser for the "analyze" command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a tennis video')
    analyze_parser.add_argument('video_path', help='Path to the input video file')
    analyze_parser.add_argument('--config', default='config/default_config.json', 
                      help='Path to the configuration file')
    analyze_parser.add_argument('--output-dir', help='Directory to save output')
    analyze_parser.add_argument('--no-visualization', action='store_true', 
                      help='Disable visualization generation')
    
    # Parser for the "prepare-data" command
    prepare_parser = subparsers.add_parser('prepare-data', 
                                     help='Prepare training data from videos')
    prepare_parser.add_argument('--config', default='config/data_preparation_config.json',
                         help='Path to the data preparation configuration file')
    prepare_parser.add_argument('--videos-dir', required=True,
                         help='Directory containing videos to process')
    prepare_parser.add_argument('--annotations-dir', required=True,
                         help='Directory containing annotation files')
    prepare_parser.add_argument('--output-dir', required=True,
                         help='Directory to save prepared data')
    
    # Parser for the "train" command
    train_parser = subparsers.add_parser('train', help='Train the RNN classifier')
    train_parser.add_argument('--config', default='config/default_config.json',
                       help='Path to the configuration file')
    train_parser.add_argument('--data-dir', required=True,
                       help='Directory containing prepared training data')
    train_parser.add_argument('--output-model', help='Path to save the trained model')
    
    return parser.parse_args()

def analyze_video(args):
    """Run the analysis pipeline on a video."""
    try:
        logger.info(f"Analyzing video: {args.video_path}")
        
        # Ensure the video file exists
        if not os.path.isfile(args.video_path):
            logger.error(f"Video file not found: {args.video_path}")
            return 1
        
        # Initialize the pipeline coordinator
        coordinator = PipelineCoordinator(args.config)
        
        # Process the video
        results = coordinator.process_video(
            video_path=args.video_path,
            output_dir=args.output_dir,
            generate_visualization=not args.no_visualization
        )
        
        logger.info(f"Analysis completed in {results['processing_time']:.2f} seconds")
        logger.info(f"Results saved to {results['output_dir']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}", exc_info=True)
        return 1

def prepare_data(args):
    """Prepare training data from videos and annotations."""
    try:
        logger.info(f"Preparing training data from {args.videos_dir}")
        
        # Ensure directories exist
        if not os.path.isdir(args.videos_dir):
            logger.error(f"Videos directory not found: {args.videos_dir}")
            return 1
            
        if not os.path.isdir(args.annotations_dir):
            logger.error(f"Annotations directory not found: {args.annotations_dir}")
            return 1
            
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        # Initialize the data preparation
        data_prep = DataPreparation(config)
        
        # Process videos and create training data
        data_prep.process_directory(
            videos_dir=args.videos_dir,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir
        )
        
        logger.info(f"Data preparation completed. Results saved to {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}", exc_info=True)
        return 1

def train_classifier(args):
    """Train the RNN classifier on prepared data."""
    try:
        logger.info(f"Training classifier using data from {args.data_dir}")
        
        # Ensure data directory exists
        if not os.path.isdir(args.data_dir):
            logger.error(f"Data directory not found: {args.data_dir}")
            return 1
            
        # Initialize the pipeline coordinator
        coordinator = PipelineCoordinator(args.config)
        
        # Train the classifier
        results = coordinator.train_classifier(
            data_dir=args.data_dir,
            output_model_path=args.output_model
        )
        
        logger.info(f"Training completed. Model saved to {results['model_path']}")
        logger.info(f"Validation accuracy: {results['evaluation_results']['accuracy']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error training classifier: {e}", exc_info=True)
        return 1

def main():
    """Main entry point for the TennisFlow application."""
    args = parse_args()
    
    # Execute the selected command
    if args.command == 'analyze':
        return analyze_video(args)
    elif args.command == 'prepare-data':
        return prepare_data(args)
    elif args.command == 'train':
        return train_classifier(args)
    else:
        logger.error("No command specified. Use one of: analyze, prepare-data, train")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 