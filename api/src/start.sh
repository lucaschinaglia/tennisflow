#!/bin/bash

# Load environment variables
set -a
source ../.env
set +a

# Check if running the API server or worker
if [ "$1" = "api" ]; then
    echo "Starting API server..."
    uvicorn main:app --host 0.0.0.0 --port ${API_PORT:-8000} --reload
elif [ "$1" = "worker" ]; then
    echo "Starting worker..."
    python worker.py
else
    echo "Usage: $0 [api|worker]"
    echo "  api    - Start the API server"
    echo "  worker - Start the worker process"
    exit 1
fi