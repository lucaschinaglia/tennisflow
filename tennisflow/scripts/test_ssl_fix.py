#!/usr/bin/env python3

import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ssl_fix_tester")

def test_ssl_fix():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Add project root to Python path if not already there
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import SSL-relevant packages first
        import ssl
        
        # Check if we already have SSL verification disabled
        context = ssl._create_default_https_context()
        is_unverified = isinstance(context, ssl._SSLContext) and not context.verify_mode
        
        if is_unverified:
            logger.info("SSL verification is already disabled")
        else:
            logger.info("Disabling SSL verification")
            ssl._create_default_https_context = ssl._create_unverified_context
        
        # Try to load the MoveNet model directly via TensorFlow Hub
        logger.info("Attempting to load MoveNet model via TensorFlow Hub")
        import tensorflow_hub as hub
        
        model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        logger.info(f"Loading model from {model_url}")
        
        model = hub.load(model_url)
        logger.info("Model loaded successfully!")
        
        # Try to use the model to ensure it's working
        logger.info("Testing model with a simple input")
        import tensorflow as tf
        import numpy as np
        
        # Check input requirements
        movenet = model.signatures['serving_default']
        logger.info(f"Model input shape requirements: {movenet.structured_input_signature}")
        
        # Get input details
        input_details = movenet.structured_input_signature[1]
        input_name = list(input_details.keys())[0]
        input_shape = input_details[input_name].shape
        input_dtype = input_details[input_name].dtype
        
        logger.info(f"Expected input: {input_name} with shape {input_shape} and dtype {input_dtype}")
        
        # Create a dummy input with correct shape and dtype
        dummy_input = tf.zeros(input_shape, dtype=input_dtype)
        
        # Run inference
        logger.info(f"Running inference with dummy input (shape: {dummy_input.shape}, dtype: {dummy_input.dtype})")
        
        # Call the model with the correct signature
        outputs = movenet(dummy_input)
        
        # Check if we have outputs
        if len(outputs) > 0:
            logger.info(f"Model inference successful! Got outputs: {list(outputs.keys())}")
            logger.info("SSL fix is working correctly!")
            return True
        else:
            logger.error("Model inference did not produce any outputs")
            return False
        
    except Exception as e:
        logger.error(f"Error testing SSL fix: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_ssl_fix()
    
    if success:
        print("\n✅ SSL fix is working correctly! You can now use MoveNet without SSL issues.")
        return True
    else:
        print("\n❌ SSL fix test failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 