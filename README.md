# TennisFlow

A mobile app for tennis players to analyze and improve their technique using machine learning.

## Overview

TennisFlow allows tennis players to record their strokes, upload videos, and receive detailed analysis and feedback to improve their game. Using advanced machine learning algorithms, the app identifies key aspects of tennis technique and provides actionable insights.

## Features

- **Video Recording & Analysis**: Record tennis strokes directly in the app or upload existing videos
- **ML-Powered Feedback**: Get detailed analysis of your technique
- **Performance Tracking**: Monitor your progress over time
- **Shot Placement Analysis**: Visualize where your shots land on the court
- **Match Analysis**: Break down your performance in match situations

## Tech Stack

- React Native / Expo
- TypeScript
- Supabase (Authentication, Database, Storage)
- Machine Learning integration for video analysis

## Development

### Prerequisites

- Node.js (v18 or newer)
- Expo CLI
- Supabase account
- iOS device with Expo Go app (for testing)

### Getting Started

1. Clone the repository
```bash
git clone https://github.com/lucaschinaglia/tennisflow.git
cd tennisflow/expo-app
```

2. Install dependencies
```bash
npm install
```

3. Start the development server
```bash
npx expo start
```

4. Scan the QR code with your iOS device's camera to open in Expo Go

## Project Structure

- `/expo-app` - React Native/Expo application
  - `/components` - Reusable UI components
  - `/screens` - App screens
  - `/hooks` - Custom React hooks
  - `/services` - API services and utilities
  - `/assets` - Images, fonts, and other static assets

## License

MIT