#!/usr/bin/env python3

import os
import sys
import ssl
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ssl_issue_fixer")

def fix_ssl_verification():
    """Apply a runtime fix to disable SSL verification globally."""
    try:
        # Backup the original context
        logger.info("Backing up original SSL context")
        original_context = ssl._create_default_https_context
        
        # Apply the unverified context
        logger.info("Applying SSL verification bypass")
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Test the fix with a simple HTTPS request
        logger.info("Testing SSL bypass with a test request")
        import urllib.request
        test_url = "https://www.google.com"
        
        try:
            with urllib.request.urlopen(test_url) as response:
                if response.status == 200:
                    logger.info("SSL bypass test successful")
                else:
                    logger.warning(f"SSL bypass test returned status code {response.status}")
        except Exception as e:
            logger.error(f"SSL bypass test failed: {e}")
            return False
        
        # Create a wrapper script to apply this fix automatically
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ssl_patch_file = os.path.join(project_root, "src", "pose_estimation", "ssl_patch.py")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(ssl_patch_file), exist_ok=True)
        
        logger.info(f"Creating SSL patch module at {ssl_patch_file}")
        
        with open(ssl_patch_file, 'w') as f:
            f.write("""# SSL verification patch
import ssl

# Disable SSL verification for the entire application
ssl._create_default_https_context = ssl._create_unverified_context
""")
        
        # Update the pose_estimation/__init__.py file to import the patch
        init_file = os.path.join(project_root, "src", "pose_estimation", "__init__.py")
        
        # Check if the file exists
        if not os.path.exists(init_file):
            logger.info(f"Creating init file at {init_file}")
            with open(init_file, 'w') as f:
                f.write("""# Import SSL patch first to ensure SSL verification is disabled
try:
    from . import ssl_patch
except ImportError:
    pass
""")
        else:
            # Read the file
            with open(init_file, 'r') as f:
                content = f.read()
            
            # Check if the patch is already imported
            if "ssl_patch" not in content:
                logger.info(f"Updating init file at {init_file}")
                
                # Add the import at the beginning
                with open(init_file, 'w') as f:
                    f.write("""# Import SSL patch first to ensure SSL verification is disabled
try:
    from . import ssl_patch
except ImportError:
    pass

""" + content)
            else:
                logger.info("SSL patch already imported in __init__.py")
        
        logger.info("SSL verification bypass has been successfully applied")
        logger.info("The TensorFlow Hub models should now download without SSL verification issues")
        
        # Let the user know how to test
        print("\nTo test the SSL fix, try running your script that uses the MoveNet pose estimator")
        print("The SSL verification should now be disabled and the model should download successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error applying SSL fix: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fix SSL verification issues by applying a runtime patch')
    
    args = parser.parse_args()
    
    success = fix_ssl_verification()
    
    if success:
        logger.info("SSL issue fix completed successfully!")
    else:
        logger.error("Failed to apply SSL fix")
        sys.exit(1)

if __name__ == "__main__":
    main() 