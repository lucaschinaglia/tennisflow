#!/usr/bin/env python3

import os
import sys
import ssl
import logging
import importlib.util
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ssl_fixer")

def apply_ssl_fix():
    """Apply a global SSL verification bypass"""
    try:
        logger.info("Applying SSL verification bypass")
        
        # Disable SSL verification globally
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Set no verify environment variable for urllib
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        
        # Also try to set it for TensorFlow Hub
        os.environ['TFHUB_VERIFY_CERT'] = '0'
        
        logger.info("SSL verification bypass applied")
        return True
    except Exception as e:
        logger.error(f"Error applying SSL fix: {e}")
        return False

def run_script(script_path, args=None):
    """Run a Python script with SSL verification disabled"""
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Apply SSL fix
    success = apply_ssl_fix()
    if not success:
        logger.warning("SSL fix could not be applied, but will try to continue")
    
    try:
        # Two approaches:
        # 1. Run as a subprocess with env vars set
        # 2. Import and run directly
        
        # Method 1: Subprocess approach
        if args is None:
            args = []
        
        logger.info(f"Running script as subprocess: {script_path} {' '.join(args)}")
        
        cmd = [sys.executable, script_path] + args
        env = os.environ.copy()
        env['PYTHONHTTPSVERIFY'] = '0'
        env['TFHUB_VERIFY_CERT'] = '0'
        
        result = subprocess.run(cmd, env=env)
        return result.returncode == 0
        
        # Method 2: Import and run directly (commented out for now)
        """
        logger.info(f"Importing and running script directly: {script_path}")
        
        # Add script's directory to path
        script_dir = os.path.dirname(os.path.abspath(script_path))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            
        # Import the script as a module
        spec = importlib.util.spec_from_file_location("target_script", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # If it has a main function, call it
        if hasattr(module, 'main'):
            return module.main()
        
        logger.warning("Script has no main() function, could not execute directly")
        return False
        """
        
    except Exception as e:
        logger.error(f"Error running script: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: run_with_ssl_fix.py <script_to_run> [script_args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    args = sys.argv[2:]
    
    success = run_script(script_path, args)
    
    if success:
        logger.info("Script executed successfully with SSL fix applied")
        sys.exit(0)
    else:
        logger.error("Script execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 