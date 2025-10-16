# Woof Mobile App

React Native mobile application for iOS and Android, built with Expo.

## Quick Start

```bash
# Install dependencies
pnpm install

# Start development server
pnpm start

# Run on iOS simulator (Mac only)
pnpm ios

# Run on Android emulator
pnpm android

# Run in web browser
pnpm web
```

## Requirements

- Node.js 18+
- pnpm 8+
- Expo CLI: `npm install -g expo-cli`
- iOS: Xcode (Mac only)
- Android: Android Studio

## Project Structure

```
src/
├── api/          # API client and endpoints
├── components/   # Reusable UI components
├── contexts/     # React Context (Auth, etc.)
├── navigation/   # React Navigation setup
├── screens/      # Screen components
├── hooks/        # Custom hooks
├── utils/        # Utility functions
└── types/        # TypeScript types
```

## Features

✅ **Implemented**:
- Authentication (Login/Register)
- JWT token management with auto-refresh
- Secure token storage
- API integration with backend
- Navigation (Auth vs Main flow)

🚧 **Coming Soon**:
- Pet profiles
- Map view with nearby pets
- Community events
- Real-time chat
- Proactive nudges
- Co-activity tracking

## Configuration

API endpoint configured in `app.json`:
```json
{
  "extra": {
    "apiUrl": "http://localhost:4000"
  }
}
```

For physical device testing, change `localhost` to your computer's local IP address.

## Native Builds

### Development Build
```bash
eas build --profile development --platform all
```

### Production Build
```bash
eas build --profile production --platform all
```

### Submit to Stores
```bash
eas submit --platform ios
eas submit --platform android
```

## Documentation

See [MOBILE_APP_SETUP.md](../../MOBILE_APP_SETUP.md) for comprehensive setup and deployment guide.

## Tech Stack

- React Native 0.81
- Expo SDK 54
- React Navigation 7
- TypeScript 5.9
- Axios for API calls
- expo-secure-store for token storage

## Support

- Expo Docs: https://docs.expo.dev
- React Navigation: https://reactnavigation.org
- Project Docs: See root MOBILE_APP_SETUP.md
