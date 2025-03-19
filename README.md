# TennisFlow - mobile app for tennis swing analysis
# TennisFlow

TennisFlow is a comprehensive tennis swing analysis app that uses computer vision and AI to help players improve their technique. The app allows users to record and upload videos of their tennis swings, which are then analyzed to provide detailed feedback and insights.

## Project Structure

The project consists of three main components:

1. **Mobile App** (`expo-app/`): An Expo-based React Native application for iOS and Android
2. **API Service** (`api/`): A FastAPI backend that handles video processing using OpenPose/MediaPipe
3. **Database** (Supabase): Cloud database and storage for user data and videos

## Features

- **Video Recording & Upload**: Capture tennis swings with your device camera or upload existing videos
- **AI-Powered Analysis**: Get detailed analysis of your tennis technique using computer vision
- **Frame-by-Frame Visualization**: See pose detection overlays with annotations
- **Performance Metrics**: Track key metrics like racket speed, hip rotation, and more
- **Improvement Suggestions**: Receive personalized feedback to improve your technique
- **Progress Tracking**: Monitor your improvement over time

## Technologies Used

- **Frontend**: React Native, Expo, TypeScript
- **Backend**: FastAPI, Redis, Docker
- **Computer Vision**: OpenPose, MediaPipe
- **Database & Storage**: Supabase
- **DevOps**: Docker Compose, GitHub Actions

## Getting Started

### Prerequisites

- Node.js (v14+)
- Python 3.8+
- Docker and Docker Compose
- Supabase account
- Expo account (for mobile app deployment)

### Mobile App Setup

```bash
# Navigate to the app directory
cd expo-app

# Install dependencies
npm install

# Start the development server
npx expo start
```

For deployment instructions, see [expo-app/DEPLOY.md](expo-app/DEPLOY.md).

### API Service Setup

```bash
# Navigate to the API directory
cd api

# Copy environment file
cp .env.example .env

# Edit .env with your credentials
# SUPABASE_URL=your_supabase_url
# SUPABASE_KEY=your_supabase_service_role_key

# Start the services with Docker Compose
docker-compose up -d
```

For detailed API deployment instructions, see [api/DEPLOY.md](api/DEPLOY.md).

### Database Setup

1. Create a new project in [Supabase](https://supabase.com/)
2. Run the SQL scripts from `supabase/schema.sql` in the SQL Editor
3. Create a storage bucket named `tennis-videos`
4. Update your `.env` files with the Supabase credentials

## Development Workflow

1. **Local Development**:
   - Run the mobile app with `npx expo start`
   - Run the API service with `docker-compose up`

2. **Testing**:
   - Build a development client: `eas build --profile development`
   - Run API tests: `cd api && pytest`

3. **Deployment**:
   - Mobile app: Follow [expo-app/DEPLOY.md](expo-app/DEPLOY.md)
   - API service: Follow [api/DEPLOY.md](api/DEPLOY.md)

## API Documentation

Once the API service is running, access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment Architecture

### Mobile App

The mobile app is built and deployed using Expo Application Services (EAS):
- Development builds for testing
- Production builds for App Store and Google Play

### API Service

The API service uses a Docker-based architecture:
- **FastAPI Server**: Handles API requests, manages uploads, and provides results
- **Worker Process**: Analyzes videos using OpenPose/MediaPipe
- **Redis Queue**: Coordinates tasks between the API and worker
- **Nginx**: (Optional) Acts as a reverse proxy in production

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## License

This project is part of the TennisFlow application.

## Contact

Lucas Chinaglia - [GitHub Profile](https://github.com/lucaschinaglia)