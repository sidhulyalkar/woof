# Woof Mobile

Native iOS/Android client for Woof, built with React Native and Expo.

The mobile product is optimized for **in-the-moment dog life** rather than mirroring every Web screen. Its primary relationship loop is:

```text
Today -> do something together -> record dog + owner outcome -> learn -> Story
```

## Current primary navigation

- **Today** — one relationship-first Adventure recommendation, why it fits, safe stop, and outcome capture.
- **Compass** — recent pathway/rhythm context from the canonical Adventure dashboard.
- **Story** — bounded relationship memory from the canonical Story API.
- **Community** — maintained social feed/reactions plus contextual Events/Nearby entry.

Contextual tools such as Daily Signals, Pets, Goals, Library, Events, Map, and Profile remain reachable without competing as permanent primary tabs.

## Canonical server integrations

The native client now consumes maintained dogOS authority directly:

- authentication/session transport;
- Adventure Today + quest selection/completion;
- Story;
- household context;
- Daily Signals capture;
- pets;
- goals;
- events;
- media library;
- social feed/reactions.

The client must not invent server protocols. For example, current auth uses the API's expiring access-token/session behavior; mobile does not manufacture a refresh-token endpoint that the server does not expose.

## Quick start

```bash
pnpm install
pnpm --filter @woof/mobile start
```

On macOS with Xcode:

```bash
pnpm --filter @woof/mobile ios
```

For a physical development device, provide an API origin reachable from that device:

```bash
EXPO_PUBLIC_API_URL=http://<your-lan-ip>:4000/api/v1 pnpm --filter @woof/mobile start
```

`EXPO_PUBLIC_API_URL` takes precedence over the development fallback in Expo config.

## Quality gates

```bash
pnpm --filter @woof/mobile type-check
pnpm --filter @woof/mobile lint
python3 .github/scripts/assert-mobile-dogos-convergence.py
```

The dedicated `dogOS Mobile Convergence CI` lane owns these checks for mobile convergence releases.

## Native build configuration

The repository contains EAS build profiles for development, preview, and production:

```bash
eas build --profile development --platform ios
eas build --profile preview --platform ios
eas build --profile production --platform ios
```

Do **not** interpret the presence of these commands as App Store readiness. The checked-in Expo config still requires environment-owned release setup such as the real EAS project identity, production API configuration, signing/provisioning, and current Apple submission qualification.

## iOS launch evidence boundary

Repository/type/lint qualification is not physical-device evidence.

Before public iOS launch, Woof still needs at minimum:

1. real EAS project/signing authority;
2. signed preview/TestFlight build against the intended production/staging API;
3. native core-loop E2E evidence;
4. at least one physical iPhone journey;
5. current Apple SDK/toolchain qualification;
6. App Privacy/privacy-manifest review for Woof and included SDKs;
7. in-app account deletion verification;
8. notification permission/delivery lifecycle qualification if native Push is included;
9. App Store metadata/screenshots/review credentials/support/privacy URLs;
10. small real-owner pilot evidence.

See [`../../docs/IOS_DOGOS_CONVERGENCE_V1.md`](../../docs/IOS_DOGOS_CONVERGENCE_V1.md).

## Project structure

```text
src/
├── api/          # Canonical API clients
├── components/   # Reusable native UI
├── contexts/     # Auth/session presentation context
├── navigation/   # Relationship-first navigation spine
├── screens/      # Native product surfaces
├── media/        # Platform media adapters
├── theme/        # Design tokens
├── utils/        # Native utilities
└── types/        # Mobile presentation types
```

## Design boundary

Woof mobile should make it easier to pay attention to the dog, not the screen.

The target interaction is deliberately simple:

> Open Woof, see one useful thing to do, put the phone away, come back to record what happened, and let the next suggestion get a little better.
