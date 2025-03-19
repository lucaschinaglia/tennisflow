#!/bin/bash

# Force use web-only mode to bypass the iOS simulator error
echo "Starting Expo without iOS simulator..."
# Add stronger environment variables to disable iOS simulator features
export EXPO_NO_IOS=1
export EXPO_NO_DEVICES=1

# Start with web only to avoid iOS simulator errors
npx expo start --web 