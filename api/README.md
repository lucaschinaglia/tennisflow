# TennisFlow API & Processing Service

This directory contains the API and processing service for the TennisFlow application. It handles video analysis for tennis swings using computer vision techniques.

## Architecture

The service is divided into several components:

1. **FastAPI Server** - Handles API requests, manages uploads, and provides analysis results.
2. **Processing Worker** - Analyzes videos using OpenPose/MediaPipe for pose estimation.
3. **Redis Queue** - Coordinates tasks between the API and worker.
4. **Supabase Integration** - Stores videos and analysis results.

## Technologies Used

- FastAPI for the API layer
- OpenPose for precise pose estimation
- MediaPipe as a fallback pose detector
- Redis for task queue management
- Docker for containerization
- Supabase for storage and database

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Supabase account with a project set up

### Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your Supabase credentials
3. Build and start the Docker services:

```bash
cd api
docker-compose up --build
```

### Environment Variables

See `.env.example` for all required environment variables.

## API Endpoints

- `POST /upload` - Upload a video for analysis
- `POST /analyze` - Analyze an already uploaded video
- `GET /status/{task_id}` - Check analysis status
- `GET /video/{video_id}/analysis` - Get analysis results
- `GET /health` - Check API health

## Development

To run the services locally outside of Docker:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API:
```bash
cd src
./start.sh api
```

3. Start the worker:
```bash
cd src
./start.sh worker
```

## OpenPose Integration

The service uses OpenPose for precise pose estimation when available. If OpenPose is not installed, it falls back to MediaPipe for basic pose detection.

To use OpenPose:
1. Set the `OPENPOSE_BIN_PATH` and `OPENPOSE_MODELS_PATH` in your `.env` file
2. The Docker setup will install OpenPose automatically

## Processing Pipeline

1. Video is uploaded to Supabase storage
2. API creates a task in Redis queue
3. Worker picks up the task and processes the video:
   - Extracts frames
   - Detects poses with OpenPose/MediaPipe
   - Analyzes swing mechanics
   - Generates metrics and annotations
4. Results are stored in Supabase
5. Mobile app can fetch and display results

## License

This project is part of the TennisFlow application.