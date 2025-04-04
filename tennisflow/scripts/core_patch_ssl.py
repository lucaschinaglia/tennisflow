#!/usr/bin/env python3

"""
Core SSL Module Patcher

This script provides a very aggressive approach to SSL verification issues by
patching the core SSL module. It should be used as a last resort when other
methods don't work.

CAUTION: This disables ALL SSL verification, which reduces security.
Only use this on your local machine for development purposes.
"""

import sys
import os
import ssl
import urllib.request
import importlib

def disable_ssl():
    """
    Apply aggressive SSL verification disabling at the Python core level.
    This approach directly patches ssl and other modules.
    """
    # Create an unverified context
    unverified_context = ssl._create_unverified_context()
    
    # Make sure it doesn't check hostname
    unverified_context.check_hostname = False
    
    # 1. Patch the ssl module
    ssl._create_default_https_context = lambda: unverified_context
    
    # 2. Patch urllib.request
    urllib.request.ssl._create_default_https_context = lambda: unverified_context
    
    # 3. Disable certificate checks completely by patching SSLContext creation
    original_create_default_context = ssl.create_default_context
    
    def patched_create_default_context(*args, **kwargs):
        context = original_create_default_context(*args, **kwargs)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    
    ssl.create_default_context = patched_create_default_context
    
    # 4. Set environment variables that might be used
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['SSL_CERT_FILE'] = '/dev/null'  # Non-existent cert file
    os.environ['REQUESTS_CA_BUNDLE'] = '/dev/null'
    
    # 5. For TensorFlow Hub specifically
    os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser("~/.tensorflow_hub")
    
    print("Core SSL verification has been disabled completely.")
    return True

def patch_modules():
    """Patch any modules that might be already loaded"""
    # Try to patch requests if it's loaded
    if 'requests' in sys.modules:
        import requests
        # Suppress SSL warnings
        requests.packages.urllib3.disable_warnings()
        
        # Disable verification in all future requests
        session = requests.Session()
        session.verify = False
        requests.Session = lambda: session
        
        print("Patched requests module to disable SSL verification.")
    
    # Try to patch urllib3 if it's loaded
    if 'urllib3' in sys.modules:
        import urllib3
        urllib3.disable_warnings()
        print("Patched urllib3 to disable SSL warnings.")
    
    # Try to patch tensorflow_hub if it's loaded
    if 'tensorflow_hub' in sys.modules:
        try:
            import tensorflow_hub as hub
            from tensorflow_hub.resolver import HttpResolverBase
            
            # Create a completely unverified context
            unverified_context = ssl._create_unverified_context()
            unverified_context.check_hostname = False
            unverified_context.verify_mode = ssl.CERT_NONE
            
            # Patch all instances of HttpResolverBase
            for obj in [getattr(hub, attr) for attr in dir(hub) if hasattr(getattr(hub, attr), '_context')]:
                obj._context = unverified_context
            
            # Directly patch the HttpResolverBase class
            # This will affect new instances
            if hasattr(HttpResolverBase, '_context'):
                HttpResolverBase._context = unverified_context
            
            # Patch the _call_urlopen method in HttpResolverBase
            original_call_urlopen = HttpResolverBase._call_urlopen
            
            def patched_call_urlopen(self, request):
                return urllib.request.urlopen(request, context=unverified_context)
            
            HttpResolverBase._call_urlopen = patched_call_urlopen
            
            print("Patched tensorflow_hub to disable SSL verification.")
        except Exception as e:
            print(f"Warning: Failed to patch tensorflow_hub: {e}")

def run_script_with_patch(script_path, args=None):
    """Run a script with SSL verification disabled"""
    if args is None:
        args = []
    
    # First apply the patches
    disable_ssl()
    patch_modules()
    
    print(f"Running script with SSL verification disabled: {script_path}")
    
    # Set the script path
    sys.argv = [script_path] + args
    
    # Execute the script
    with open(script_path, 'rb') as f:
        script_content = f.read()
    
    # Run the script in the current process
    # This ensures all our patches are active
    compiled = compile(script_content, script_path, 'exec')
    
    # Set up globals with our patched SSL
    globals_dict = {
        '__name__': '__main__', 
        '__file__': script_path,
        'ssl': ssl  # Provide our patched ssl module
    }
    
    exec(compiled, globals_dict)

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python core_patch_ssl.py <script.py> [args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    args = sys.argv[2:]
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)
    
    try:
        run_script_with_patch(script_path, args)
    except Exception as e:
        print(f"Error running script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 