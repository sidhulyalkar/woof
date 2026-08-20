# 🚀 Woof Mobile - Quick Start

## Prerequisites
- Node.js 18+
- pnpm 8+
- Expo CLI: `npm install -g expo-cli`

## Start Developing (3 Steps)

### 1. Start API
```bash
cd apps/api
pnpm dev
```
API runs on http://localhost:4000

### 2. Start Mobile
```bash
cd apps/mobile
pnpm start
```

### 3. Run App
**iOS** (Mac only):
```bash
pnpm ios
```

**Android**:
```bash
pnpm android
```

**Physical Device**:
1. Install Expo Go app
2. Scan QR code

## Test Authentication

**Register a new user:**
- Display Name: Test User
- Handle: @testuser
- Email: test@example.com
- Password: password123

**Or use existing:**
- Email: sarah@example.com
- Password: password123

## Project Structure

```
apps/mobile/src/
├── api/         # API client & endpoints
├── screens/     # App screens (Login, Register, Home)
├── contexts/    # Auth state management
├── navigation/  # React Navigation
└── components/  # Reusable components (add yours here!)
```

## Add a New Screen

1. Create `src/screens/YourScreen.tsx`
2. Add to `src/navigation/AppNavigator.tsx`
3. Use API client: `import apiClient from '@/api/client'`

## Useful Commands

```bash
# Clear cache
pnpm start --clear

# Check TypeScript
npx tsc --noEmit

# Build for iOS
eas build --profile production --platform ios

# Build for Android
eas build --profile production --platform android
```

## Documentation

📖 **Full Guide**: [MOBILE_APP_SETUP.md](./MOBILE_APP_SETUP.md)
📊 **Status**: [MOBILE_STATUS.md](./MOBILE_STATUS.md)
📝 **Summary**: [MOBILE_APP_SUMMARY.md](./MOBILE_APP_SUMMARY.md)

## Need Help?

1. Check [MOBILE_APP_SETUP.md](./MOBILE_APP_SETUP.md) troubleshooting section
2. Expo docs: https://docs.expo.dev
3. React Navigation: https://reactnavigation.org

---

**Status**: ✅ Ready to build features!
