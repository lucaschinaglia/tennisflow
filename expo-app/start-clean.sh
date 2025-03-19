#!/bin/bash

echo "Cleaning up any existing Metro/Expo processes..."
pkill -f "metro" || true
pkill -f "expo" || true

echo "Clearing Metro cache..."
rm -rf node_modules/.cache

echo "Ensuring correct React types are installed..."
rm -rf node_modules/@types/react
npm install @types/react@~18.3.12 --no-save

echo "Starting Expo app with web support..."
npx expo start --web 