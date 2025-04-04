#!/usr/bin/env python3
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
