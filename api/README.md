# TennisFlow API Service

This directory contains the backend ML processing API for TennisFlow. The API handles video analysis, processing, and ML model integration.

## Architecture

The TennisFlow API service follows a serverless architecture with three main components:

1. **API Gateway**: Handles HTTP requests and authentication
2. **Processing Queue**: Manages the video analysis job queue
3. **ML Processing Service**: Runs the ML models on the videos

## Flow

1. Mobile app uploads a video to Supabase Storage
2. App calls the API to request analysis of the video
3. API adds the video to the processing queue
4. ML service takes videos from the queue, processes them, and updates the database with results
5. App polls for status updates or receives push notifications when processing is complete

## ML Components

The ML pipeline consists of several components:

- **Pose Estimation**: Detects and tracks body key points using models like PoseNet or BlazePose
- **Stroke Classification**: Identifies the type of tennis stroke (forehand, backhand, serve, etc.)
- **Technique Analysis**: Analyzes swing mechanics and body positioning
- **Performance Metrics**: Calculates metrics like racket speed, hip rotation, etc.
- **Feedback Generation**: Uses rule-based and ML systems to generate actionable feedback

## Deployment

The API components are deployed using:

- AWS Lambda for serverless functions
- AWS SQS for queue management
- AWS S3 for temporary storage
- TensorFlow Serving for ML model hosting

## Local Development

For local development, you can use the mock API service included in this directory:

```bash
# Install dependencies
npm install

# Start local development server
npm run dev
```

## API Endpoints

The API exposes the following endpoints:

- `POST /videos/analyze` - Request analysis of a video
- `GET /videos/{id}/status` - Check the status of video processing
- `GET /videos/{id}/results` - Get the analysis results for a video
- `POST /videos/{id}/feedback` - Add feedback to a video
- `GET /users/{id}/progress` - Get user progress metrics

## ML Model Development

The models are trained using TensorFlow and PyTorch. The training data consists of annotated tennis videos from professional players and amateur players.

Model variants:
- High-accuracy model (used for cloud processing)
- Lightweight model (can run on-device for basic analysis)

## Future Improvements

- Real-time analysis for immediate feedback
- On-device ML for faster processing
- Coach-to-player feedback system
- Comparative analysis between players