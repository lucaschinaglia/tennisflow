# TennisFlow Mobile App Deployment Guide

This guide explains how to build and deploy the TennisFlow mobile app using Expo Application Services (EAS).

## Prerequisites

- Node.js (v14+)
- Expo CLI (`npm install -g expo-cli eas-cli`)
- Expo account (sign up at [expo.dev](https://expo.dev))
- Apple Developer account (for iOS deployment)
- Google Play Developer account (for Android deployment)

## Setup

### 1. Install Dependencies

```bash
cd expo-app
npm install
```

### 2. Configure Environment Variables

The app uses environment variables for configuration which are set in the `eas.json` file. Update the following variables as needed:

- `EXPO_PUBLIC_SUPABASE_URL`: Your Supabase project URL
- `EXPO_PUBLIC_SUPABASE_ANON_KEY`: Your Supabase anonymous key
- `EXPO_PUBLIC_API_URL`: URL of your deployed API server

### 3. Log in to Expo

```bash
eas login
```

### 4. Configure App Configuration

Update `app.json` to customize your app's name, icons, and other settings.

## Development

### Run in Development Mode

```bash
npx expo start
```

### Build Development Client

```bash
eas build --profile development --platform [android|ios]
```

## Testing

### Build Preview Version

This creates a build for internal testing:

```bash
# For Android APK
eas build --profile preview --platform android

# For iOS simulator build
eas build --profile preview --platform ios
```

## Production Deployment

### 1. Configure App Store / Play Store Information

For iOS, you'll need to:
- Create an app in App Store Connect
- Obtain an App Store ID
- Configure your Apple Team ID

For Android, you'll need to:
- Create an app in Google Play Console
- Generate a Google Play API key

### 2. Update `eas.json` Submission Configuration

Update the `submit` section of `eas.json`:

```json
"submit": {
  "production": {
    "ios": {
      "appleId": "YOUR_APPLE_ID",
      "ascAppId": "YOUR_APP_STORE_CONNECT_APP_ID",
      "appleTeamId": "YOUR_APPLE_TEAM_ID"
    },
    "android": {
      "serviceAccountKeyPath": "path/to/api-key.json",
      "track": "production"
    }
  }
}
```

### 3. Build Production Version

```bash
# Build for both platforms
eas build --profile production --platform all

# Or build for a specific platform
eas build --profile production --platform [android|ios]
```

### 4. Submit to App Stores

```bash
# Submit to both platforms
eas submit --platform all

# Or submit to a specific platform
eas submit --platform [android|ios]
```

## Updates Using Expo Updates

Once your app is live, you can push updates without requiring new store submissions:

```bash
eas update --branch production --message "Update message"
```

This allows you to update the JavaScript bundle without requiring a new binary build.

## Environment-Specific Configurations

The `eas.json` file contains different environment configurations:

- `development`: For local development and testing
- `preview`: For internal testing builds
- `production`: For App Store/Play Store releases

## Handling Credentials

EAS manages your credentials (signing keys, provisioning profiles, etc.) and stores them securely. You can configure credentials behavior in `eas.json`:

```json
"build": {
  "production": {
    "credentialsSource": "remote" // or "local"
  }
}
```

## Troubleshooting

### Build Failures

Check the build logs provided by EAS for specific errors. Common issues include:

- Missing dependencies in `package.json`
- Incompatible native modules
- Invalid configuration in `app.json`

### Submission Failures

If app submission fails:

1. Check that your app meets the store guidelines
2. Verify your app's metadata is complete in the store console
3. Ensure your provisioning profiles and certificates are valid (iOS)
4. Verify your Google Play API key has the correct permissions (Android)

## Additional Resources

- [Expo Documentation](https://docs.expo.dev/)
- [EAS Build Docs](https://docs.expo.dev/build/introduction/)
- [EAS Submit Docs](https://docs.expo.dev/submit/introduction/)