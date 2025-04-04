#!/usr/bin/env python3

"""
Monkey patch TensorFlow Hub to disable SSL verification.

This script doesn't modify any files; it applies the patch at runtime.
Import this module at the beginning of your scripts to disable SSL verification
in TensorFlow Hub.
"""

def apply_patch():
    """Apply the monkey patch to TensorFlow Hub."""
    import ssl
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("tf_hub_patch")
    
    # First, disable SSL verification globally
    logger.info("Disabling SSL verification globally")
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        # Now patch TensorFlow Hub specifically
        import tensorflow_hub as hub
        
        # Find the resolver class that needs patching
        from tensorflow_hub.resolver import HttpResolverBase
        
        # Save the original __init__ method
        original_init = HttpResolverBase.__init__
        
        # Define our patched init method
        def patched_init(self):
            # Call the original __init__
            original_init(self)
            
            # Replace the SSL context with an unverified one
            self._context = ssl._create_unverified_context()
            
            logger.info("Patched TensorFlow Hub SSL context")
        
        # Apply our patch
        HttpResolverBase.__init__ = patched_init
        
        logger.info("TensorFlow Hub successfully patched to ignore SSL certificate verification")
        
        return True
        
    except ImportError:
        logger.warning("TensorFlow Hub not found, skipping specific patch")
        return False
    except Exception as e:
        logger.error(f"Failed to patch TensorFlow Hub: {e}")
        return False

# Apply the patch when this module is imported
success = apply_patch()

if __name__ == "__main__":
    import sys
    
    if success:
        print("TensorFlow Hub successfully patched!")
        
        # If arguments were provided, run the specified script
        if len(sys.argv) > 1:
            import subprocess
            
            script = sys.argv[1]
            args = sys.argv[2:]
            
            print(f"Running: {script}")
            cmd = [sys.executable, script] + args
            
            result = subprocess.run(cmd)
            sys.exit(result.returncode)
            
        sys.exit(0)
    else:
        print("Failed to patch TensorFlow Hub")
        sys.exit(1) 