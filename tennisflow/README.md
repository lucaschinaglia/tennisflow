# TennisFlow

TennisFlow is a tennis motion analysis system that uses computer vision and machine learning to analyze tennis swings and provide feedback on technique.

## Features

- Tennis pose classification (forehand, backhand, serve, ready position)
- Tennis event detection (serve, hit, bounce, net)
- Video analysis from file, URL, or base64 encoded data
- API for integration with web and mobile applications

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/tennisflow.git
cd tennisflow
pip install -r requirements.txt
```

## Model Training

The project includes scripts for training tennis pose and event detection models:

### Pose Model Training

```bash
cd tennisflow
python -m models.pose_analysis.train_pose_model
```

### Event Model Training

```bash
cd tennisflow
python -m models.event_detection.train_event_model
```

## Using the API

Start the FastAPI server:

```bash
cd tennisflow
python -m api.app
```

The API will be available at http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

### API Endpoints

- `GET /`: Health check
- `GET /status`: Check if the service is ready
- `POST /analyze/url`: Analyze a video from a URL
- `POST /analyze/upload`: Analyze an uploaded video file
- `POST /analyze/base64`: Analyze a base64 encoded video

## Using the Models Directly

You can also use the models directly in your Python code:

```python
from tennisflow.models.inference import TennisAnalyzer

# Create an analyzer
analyzer = TennisAnalyzer()

# Analyze a video
results = analyzer.analyze_video("path/to/video.mp4")

# Print results
print(results)
```

## Inference Module

The inference module provides mock implementation to simulate model inference for testing and development:

```python
from tennisflow.models.inference import load_pose_model, process_video

# Load a pose model
pose_model = load_pose_model("path/to/model.pth")

# Process a video
results = process_video("path/to/video.mp4", pose_model)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.