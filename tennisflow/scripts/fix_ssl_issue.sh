#!/bin/bash

# Script to fix SSL certificate issues with MoveNet model
# This is a shell wrapper for fix_ssl_issue.py

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python environment is activated
if [ -z "${VIRTUAL_ENV}" ]; then
    # Try to activate the venv if it exists
    if [ -d "${SCRIPT_DIR}/../.venv" ]; then
        echo "Activating Python virtual environment..."
        source "${SCRIPT_DIR}/../.venv/bin/activate"
    else
        echo "WARNING: No virtual environment is activated. This might cause issues if dependencies are not installed globally."
    fi
fi

# Check if the script exists
if [ ! -f "${SCRIPT_DIR}/fix_ssl_issue.py" ]; then
    echo "Error: fix_ssl_issue.py script not found at ${SCRIPT_DIR}/fix_ssl_issue.py"
    exit 1
fi

# Make sure script is executable
chmod +x "${SCRIPT_DIR}/fix_ssl_issue.py"

# Run the fix_ssl_issue.py script
python "${SCRIPT_DIR}/fix_ssl_issue.py"

# Check if the script was successful
if [ $? -eq 0 ]; then
    echo "SSL issue fix completed successfully!"
    echo "You can now run your scripts that use the MoveNet pose estimator without SSL verification issues."
else
    echo "Failed to fix SSL issue. Please check the error messages above."
    exit 1
fi 