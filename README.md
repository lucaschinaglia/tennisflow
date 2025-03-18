# TennisFlow

A comprehensive tennis swing analysis application built with React Native, Expo, and Supabase, leveraging computer vision to provide real-time feedback and improvement suggestions for tennis players.

## Project Structure

- `/expo-app`: Mobile application built with React Native and Expo
- `/supabase`: Supabase configuration, database schema, and setup instructions
- `/.github`: GitHub Actions workflows for CI/CD

## Features

- **User Authentication**: Secure sign up and login using Supabase Auth
- **Video Upload**: Record or upload tennis swing videos for analysis
- **AI-Powered Analysis**: Computer vision analysis of tennis swing mechanics
- **Detailed Feedback**: Get actionable insights on your technique
- **Progress Tracking**: Monitor improvement over time
- **Interactive UI**: User-friendly interface for viewing analysis results

## Technology Stack

- **Frontend**: React Native, Expo
- **Backend**: Supabase (PostgreSQL, Auth, Storage)
- **Database**: PostgreSQL (via Supabase)
- **Authentication**: Supabase Auth
- **File Storage**: Supabase Storage
- **CI/CD**: GitHub Actions

## Getting Started

### Prerequisites

- Node.js (v16+)
- npm or yarn
- Expo CLI
- Supabase account

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lucaschinaglia/tennisflow.git
   cd tennisflow
   ```

2. Set up Supabase:
   - Follow the instructions in `supabase/README.md`

3. Set up the Expo app:
   - Navigate to the expo app directory:
     ```bash
     cd expo-app
     ```
   - Install dependencies:
     ```bash
     npm install
     # or
     yarn install
     ```
   - Create a `.env` file with your Supabase credentials (use `.env.example` as a template)
   - Start the development server:
     ```bash
     npm start
     # or
     yarn start
     ```

## Development Roadmap

### Phase 1: Foundation and Setup
- [x] Project structure setup
- [x] GitHub repository initialization
- [x] Supabase integration
- [x] User authentication
- [x] Basic UI components

### Phase 2: Core Functionality
- [ ] Video recording and upload
- [ ] Initial video processing
- [ ] Basic swing analysis
- [ ] Results display

### Phase 3: Enhanced Analysis
- [ ] Advanced swing mechanics analysis
- [ ] Comparison with ideal form
- [ ] Personalized feedback
- [ ] Progress tracking

### Phase 4: Refinement and Release
- [ ] UI/UX polish
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] App store deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.