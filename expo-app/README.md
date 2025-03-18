# TennisFlow Mobile App

A React Native mobile application built with Expo for tennis swing analysis using computer vision.

## Features

- User authentication with Supabase
- Upload tennis videos for analysis
- View detailed swing analytics
- Personalized feedback and improvement suggestions
- Track progress over time
- User profiles

## Setup

### Prerequisites

- Node.js (v16+)
- npm or yarn
- Expo CLI
- Supabase account and project

### Environment Variables

Create a `.env` file in the root of the project with your Supabase credentials:

```
EXPO_PUBLIC_SUPABASE_URL=your_supabase_url
EXPO_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Installation

```bash
# Install dependencies
npm install
# or
yarn install

# Start the Expo development server
npm start
# or
yarn start
```

## Project Structure

- `components/` - Reusable UI components
- `screens/` - Main app screens
- `hooks/` - Custom React hooks
- `lib/` - Utility functions and API clients
- `assets/` - Static assets like images and fonts

## Database Schema

The application uses Supabase as its backend with the following tables:

- `users` - User profiles and authentication
- `videos` - Uploaded videos and analysis results

## Third-Party Libraries

- Expo
- React Navigation
- Supabase
- Expo AV for video playback
- Expo Image Picker for media selection