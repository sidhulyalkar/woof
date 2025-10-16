# Woof Mobile App Setup Guide

## Overview

The Woof mobile app is built with React Native and Expo, providing native iOS and Android applications. This guide covers setup, development, and deployment.

## Architecture

### Tech Stack
- **Framework**: React Native with Expo SDK 54
- **Navigation**: React Navigation v7
- **State Management**: React Context API (Auth)
- **API Client**: Axios with JWT token refresh
- **Secure Storage**: expo-secure-store for tokens
- **Native Features**: Camera, Location, Maps, Notifications

### Project Structure
```
apps/mobile/
├── src/
│   ├── api/              # API client and endpoints
│   │   ├── client.ts     # Axios instance with interceptors
│   │   └── auth.ts       # Authentication API calls
│   ├── components/       # Reusable UI components
│   ├── contexts/         # React Context providers
│   │   └── AuthContext.tsx
│   ├── navigation/       # Navigation configuration
│   │   └── AppNavigator.tsx
│   ├── screens/          # Screen components
│   │   ├── LoginScreen.tsx
│   │   ├── RegisterScreen.tsx
│   │   └── HomeScreen.tsx
│   ├── hooks/            # Custom React hooks
│   ├── utils/            # Utility functions
│   └── types/            # TypeScript type definitions
├── assets/               # Images, fonts, etc.
├── App.tsx               # App entry point
├── app.json              # Expo configuration
├── eas.json              # EAS Build configuration
└── package.json
```

## Prerequisites

1. **Node.js**: v18 or higher
2. **pnpm**: v8 or higher
3. **Expo CLI**: Install globally
   ```bash
   npm install -g expo-cli
   ```
4. **EAS CLI**: For native builds
   ```bash
   npm install -g eas-cli
   ```

### iOS Development (Mac only)
- **Xcode**: Latest version from App Store
- **iOS Simulator**: Included with Xcode
- **CocoaPods**: `sudo gem install cocoapods`

### Android Development
- **Android Studio**: Download from android.com
- **Android SDK**: API Level 31+
- **Android Emulator**: Create via Android Studio AVD Manager

## Installation

1. **Install Dependencies**
   ```bash
   cd /Users/sidhulyalkar/Documents/App_Dev/woof
   pnpm install
   ```

2. **Start the API Server** (required for mobile app)
   ```bash
   cd apps/api
   pnpm dev
   ```

3. **Start Mobile App**
   ```bash
   cd apps/mobile
   pnpm start
   ```

## Development

### Running on Simulators/Emulators

**iOS Simulator** (Mac only):
```bash
cd apps/mobile
pnpm ios
```

**Android Emulator**:
```bash
cd apps/mobile
pnpm android
```

**Web Browser** (for quick testing):
```bash
cd apps/mobile
pnpm web
```

### Running on Physical Devices

1. Install **Expo Go** app from App Store or Google Play
2. Start the dev server: `pnpm start`
3. Scan the QR code with:
   - **iOS**: Camera app
   - **Android**: Expo Go app

### Environment Configuration

The API URL is configured in [app.json](apps/mobile/app.json):

```json
{
  "extra": {
    "apiUrl": "http://localhost:4000"
  }
}
```

**For Physical Device Testing**:
- Replace `localhost` with your computer's local IP address
- Example: `"apiUrl": "http://192.168.1.100:4000"`
- Find your IP: `ifconfig | grep "inet " | grep -v 127.0.0.1`

### Hot Reload

Expo supports fast refresh. Changes to code will automatically reload the app.

**Clear Cache**:
```bash
pnpm start --clear
```

## Authentication Flow

The mobile app uses JWT-based authentication with automatic token refresh:

1. **Login/Register**: Tokens stored in secure storage
2. **API Requests**: Access token automatically attached to requests
3. **Token Refresh**: Automatic on 401 errors
4. **Logout**: Tokens cleared from secure storage

### API Client Features

Located in [src/api/client.ts](apps/mobile/src/api/client.ts):

- **Axios interceptors** for token management
- **Automatic token refresh** on 401 responses
- **TypeScript-first** API methods
- **Configurable base URL** via app.json

## Native Features

### Camera & Photo Library

Configured in [app.json](apps/mobile/app.json):
- Camera access for pet photos
- Photo library access for selecting images
- Permissions automatically requested on first use

### Location Services

Configured for:
- **Foreground location**: Show nearby pet parents
- **Background location**: Proximity notifications
- Permissions handled by expo-location

### Push Notifications

Setup via expo-notifications:
- Device token registration with backend
- Foreground/background notification handling
- Deep linking support

### Maps

Using react-native-maps:
- Show nearby parks and pet parents
- Real-time location tracking
- Custom markers for pets and events

## Building for Production

### EAS Build (Recommended)

EAS Build handles native compilation in the cloud.

**Initial Setup**:
```bash
cd apps/mobile
eas login
eas build:configure
```

**Development Build**:
```bash
eas build --profile development --platform ios
eas build --profile development --platform android
```

**Production Build**:
```bash
# iOS (generates .ipa)
eas build --profile production --platform ios

# Android (generates .aab for Play Store)
eas build --profile production --platform android

# Both platforms
eas build --profile production --platform all
```

### Build Profiles

Configured in [eas.json](apps/mobile/eas.json):

- **development**: For testing with dev client
- **preview**: For internal testing (APK/IPA)
- **production**: For app store submission

## Deployment

### iOS App Store

1. **Build for production**:
   ```bash
   eas build --profile production --platform ios
   ```

2. **Submit to App Store**:
   ```bash
   eas submit --platform ios
   ```

3. **Requirements**:
   - Apple Developer Account ($99/year)
   - App Store Connect setup
   - Bundle identifier: `com.woof.app`

### Google Play Store

1. **Build for production**:
   ```bash
   eas build --profile production --platform android
   ```

2. **Submit to Play Store**:
   ```bash
   eas submit --platform android
   ```

3. **Requirements**:
   - Google Play Developer Account ($25 one-time)
   - Play Console setup
   - Package name: `com.woof.app`

## Troubleshooting

### Common Issues

**Metro bundler connection failed**:
```bash
pnpm start --clear
```

**iOS build fails**:
```bash
cd ios && pod install && cd ..
```

**Android build fails**:
```bash
cd android && ./gradlew clean && cd ..
```

**API connection fails**:
- Check API is running: `curl http://localhost:4000/health`
- Update apiUrl in app.json to your local IP
- Disable firewall for port 4000

### Debugging

**Enable Remote Debugging**:
- Shake device or press `Cmd+D` (iOS) / `Cmd+M` (Android)
- Select "Debug Remote JS"

**View Logs**:
```bash
# iOS
npx react-native log-ios

# Android
npx react-native log-android
```

**Expo DevTools**:
- Runs automatically with `pnpm start`
- Access at http://localhost:19002

## Testing

### Manual Testing Checklist

- [ ] Login/Register flow
- [ ] Token refresh on 401
- [ ] Logout clears tokens
- [ ] Camera permissions
- [ ] Location permissions
- [ ] Notification permissions
- [ ] Network error handling
- [ ] Offline mode graceful handling

### Automated Testing (Future)

Add Jest and React Native Testing Library:
```bash
pnpm add -D @testing-library/react-native jest
```

## Performance

### Bundle Size Optimization

Expo automatically optimizes:
- Tree shaking unused code
- Minification in production
- Asset compression

### Monitoring

Consider adding:
- **Sentry**: Error tracking
- **Firebase Analytics**: User behavior
- **Firebase Performance**: Performance monitoring

## Security

### Best Practices

✅ **Implemented**:
- Secure token storage (expo-secure-store)
- HTTPS for API calls (production)
- Automatic token refresh
- Sensitive data not in AsyncStorage

⚠️ **TODO**:
- Add certificate pinning
- Implement biometric authentication
- Add jailbreak/root detection

## Next Steps

### Core Features to Implement

1. **Pet Profiles**
   - Create/edit pet profiles
   - Upload pet photos
   - Pet details (breed, age, etc.)

2. **Map View**
   - Show nearby pet parents
   - Real-time location updates
   - Park markers

3. **Events**
   - Browse community events
   - RSVP to events
   - Event check-in with QR code

4. **Chat**
   - Direct messaging
   - Socket.io integration
   - Push notifications for messages

5. **Nudges**
   - Receive proactive nudges
   - Accept/dismiss actions
   - Location-based triggers

6. **Co-Activity**
   - Find nearby dogs in real-time
   - Activity tracking
   - Share walk routes

### Architecture Enhancements

- [ ] Add Redux or Zustand for complex state
- [ ] Implement React Query for API caching
- [ ] Add offline-first sync with WatermelonDB
- [ ] Create shared component library with Storybook
- [ ] Add E2E tests with Detox

## Resources

- **Expo Docs**: https://docs.expo.dev
- **React Navigation**: https://reactnavigation.org
- **EAS Build**: https://docs.expo.dev/build/introduction
- **React Native**: https://reactnative.dev

## Support

For issues or questions:
1. Check this documentation
2. Review Expo documentation
3. Check API logs in `apps/api`
4. Enable debug mode in mobile app

---

**Last Updated**: October 16, 2025
**Version**: 1.0.0
**Status**: ✅ Setup Complete - Ready for Feature Development
