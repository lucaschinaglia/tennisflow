#!/usr/bin/env python3

"""
Wrapper script for compare_models_on_video.py with SSL verification fix.

This script applies SSL certificate verification bypass and then runs
the model comparison on video.

Usage:
    ./scripts/run_video_comparison.py [arguments for compare_models_on_video.py]
"""

import os
import sys
import ssl
import subprocess

def apply_ssl_fix():
    """Apply SSL verification bypass"""
    print("Applying SSL verification bypass...")
    
    # Create an unverified context
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Set environment variables
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['SSL_CERT_FILE'] = '/dev/null'
    os.environ['REQUESTS_CA_BUNDLE'] = '/dev/null'
    
    # For TensorFlow Hub specifically
    os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser("~/.tensorflow_hub")
    
    print("SSL verification bypass applied successfully")
    return True

def run_script_with_ssl_fix():
    """Run the video comparison script with SSL fix applied"""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the compare_models_on_video.py script
    compare_script = os.path.join(script_dir, "compare_models_on_video.py")
    
    if not os.path.exists(compare_script):
        print(f"Error: Video comparison script not found at {compare_script}")
        return False
    
    # Apply SSL fix
    apply_ssl_fix()
    
    # Get arguments to pass to the script
    args = sys.argv[1:]
    
    # Run the script with SSL verification disabled
    print(f"Running compare_models_on_video.py with SSL verification disabled...")
    
    # Set up the command
    cmd = [sys.executable, compare_script] + args
    
    # Run the command
    try:
        result = subprocess.run(cmd, env=os.environ)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running compare_models_on_video.py: {e}")
        return False

if __name__ == "__main__":
    success = run_script_with_ssl_fix()
    sys.exit(0 if success else 1) 