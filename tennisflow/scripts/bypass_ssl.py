#!/usr/bin/env python3

"""
Direct SSL Verification Bypass

This script provides a direct way to bypass SSL certificate verification issues in Python,
particularly for macOS users who often encounter these problems.

Usage:
1. Import this module at the very beginning of your script:
   import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))); from scripts.bypass_ssl import bypass_ssl; bypass_ssl()

2. Or run your script with this as a wrapper:
   python -c "import bypass_ssl; bypass_ssl.bypass_ssl()" your_script.py
"""

import os
import sys
import ssl
import certifi
import logging
import tempfile
import shutil
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ssl_bypass")

def bypass_ssl(mode="direct"):
    """
    Apply SSL verification bypass
    
    Args:
        mode: How to bypass SSL
            - "direct": Replace the SSL context directly (most reliable)
            - "env": Set environment variables
            - "certifi": Use certifi's certificates
            - "all": Apply all methods
    """
    logger.info(f"Applying SSL verification bypass (mode={mode})")
    
    if mode in ["direct", "all"]:
        # Method 1: Direct replacement of SSL context
        logger.info("Applying direct SSL context replacement")
        ssl._create_default_https_context = ssl._create_unverified_context
    
    if mode in ["env", "all"]:
        # Method 2: Environment variables
        logger.info("Setting environment variables for SSL bypass")
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # Specific for TensorFlow Hub
        os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser("~/.tensorflow_hub")
    
    if mode in ["certifi", "all"]:
        # Method 3: Use certifi's certificates
        logger.info("Using certifi's certificates")
        try:
            # Create a properly configured context with certifi's bundle
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl._create_default_https_context = lambda: ssl_context
        except Exception as e:
            logger.error(f"Error setting up certifi context: {e}")
    
    logger.info("SSL bypass methods applied")
    
    # Test if it worked
    test_ssl_verification()

def test_ssl_verification():
    """Test if SSL verification is working with a simple request"""
    try:
        import urllib.request
        
        logger.info("Testing SSL bypass with a request to www.google.com")
        with urllib.request.urlopen("https://www.google.com") as response:
            if response.status == 200:
                logger.info("SSL bypass test successful!")
            else:
                logger.warning(f"SSL bypass test returned status code {response.status}")
    except Exception as e:
        logger.error(f"SSL bypass test failed: {e}")

def patch_python_ssl_mac():
    """
    Attempt to fix SSL issues on macOS by installing certificates properly
    
    This is a more permanent solution for macOS users
    """
    try:
        logger.info("Attempting to fix macOS SSL verification issues")
        
        # Get certificate file from certifi
        certifi_path = certifi.where()
        logger.info(f"Using certifi certificates at: {certifi_path}")
        
        # Find Python's SSL directory
        python_path = sys.executable
        python_dir = os.path.dirname(os.path.dirname(python_path))
        
        # Different possible locations
        possible_paths = [
            os.path.join(python_dir, "Extras", "etc", "openssl", "cert.pem"),
            os.path.join(python_dir, "etc", "openssl", "cert.pem"),
            os.path.join("/etc/ssl/", "cert.pem"),
            os.path.join(python_dir, "ssl", "cert.pem"),
            os.path.join(os.path.expanduser("~"), ".ssl", "cert.pem")
        ]
        
        # Try to create directories and copy the cert.pem file
        for dest_path in possible_paths:
            try:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copyfile(certifi_path, dest_path)
                logger.info(f"Copied certificates to: {dest_path}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Couldn't copy to {dest_path}: {e}")
        
        logger.info("Certificate installation attempted. Please restart your program.")
        
    except Exception as e:
        logger.error(f"Error patching SSL certificates: {e}")

def create_wrapper_script():
    """
    Create a wrapper script that can be used to run any Python script with SSL verification disabled
    """
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        wrapper_path = os.path.join(script_dir, "ssl_wrapper.py")
        
        with open(wrapper_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
import os
import sys
import ssl
import subprocess

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Check if a script was provided
if len(sys.argv) < 2:
    print("Usage: ssl_wrapper.py <script_to_run> [args...]")
    sys.exit(1)

# Get the script and arguments
script = sys.argv[1]
args = sys.argv[2:]

# Run the script with SSL verification disabled
cmd = [sys.executable, script] + args
env = os.environ.copy()

result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)
""")
        
        # Make it executable
        os.chmod(wrapper_path, 0o755)
        
        print(f"\nWrapper script created at: {wrapper_path}")
        print("You can use it to run any script with SSL verification disabled:")
        print(f"  python {wrapper_path} your_script.py arg1 arg2 ...")
        
    except Exception as e:
        logger.error(f"Error creating wrapper script: {e}")

def patch_tensorflow_hub():
    """Attempt to patch TensorFlow Hub's code to bypass SSL verification"""
    try:
        import tensorflow_hub as hub
        
        # Find the TensorFlow Hub directory
        hub_dir = os.path.dirname(hub.__file__)
        resolver_file = os.path.join(hub_dir, "resolver.py")
        
        if not os.path.exists(resolver_file):
            logger.error(f"TensorFlow Hub resolver file not found at {resolver_file}")
            return
        
        # Read the file
        with open(resolver_file, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if "# SSL verification bypass" in content:
            logger.info("TensorFlow Hub resolver already patched")
            return
        
        # Make a backup
        backup_file = resolver_file + ".bak"
        if not os.path.exists(backup_file):
            shutil.copyfile(resolver_file, backup_file)
            logger.info(f"Created backup of resolver.py at {backup_file}")
        
        # Find the line to patch
        if "context = ssl.create_default_context()" in content:
            patched_content = content.replace(
                "context = ssl.create_default_context()",
                "# SSL verification bypass\n    context = ssl._create_unverified_context()"
            )
            
            # Write the patched file
            with open(resolver_file, 'w') as f:
                f.write(patched_content)
            
            logger.info("Successfully patched TensorFlow Hub resolver to bypass SSL verification")
        else:
            logger.warning("Could not find the line to patch in TensorFlow Hub resolver")
            
    except Exception as e:
        logger.error(f"Error patching TensorFlow Hub: {e}")

def main():
    """Main function"""
    bypass_ssl(mode="all")
    create_wrapper_script()
    patch_tensorflow_hub()
    
    if sys.platform == 'darwin':  # macOS
        patch_python_ssl_mac()
    
    print("\n========================================")
    print("SSL verification bypass applied successfully!")
    print("You can now import this module at the beginning of your scripts:")
    print("  from scripts.bypass_ssl import bypass_ssl; bypass_ssl()")
    print("Or use the wrapper script to run your programs:")
    print("  python scripts/ssl_wrapper.py your_script.py")
    print("========================================\n")

if __name__ == "__main__":
    main() 