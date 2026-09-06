# Expo SDK 54 Compatibility v1

## Purpose

Woof's native client was already configured for Expo SDK 54, but several native packages were still pinned to versions from an earlier or otherwise unsupported compatibility set. The first macOS/Xcode qualification run exposed that drift when React Native Codegen failed while resolving `react-native-screens` before an iOS workspace could compile.

This tranche aligns the managed native dependency graph to Expo SDK 54's compatibility authority and adds a permanent read-only CI gate so the repository cannot silently drift back out of that supported graph.

## Qualified alignment

The canonical mobile manifest now uses the Expo SDK 54-compatible package family selected and validated by `expo install --fix` / `expo install --check`, including:

- React Native `0.81.5`
- `react-native-screens` `^4.16.0`
- `react-native-gesture-handler` `~2.28.0`
- `react-native-maps` `1.20.1`
- `react-native-safe-area-context` `^5.6.2`
- `expo-camera` `~17.0.10`
- `expo-constants` `~18.0.14`
- `expo-image-picker` `~17.0.11`
- `expo-linking` `~8.0.12`
- `expo-location` `~19.0.8`
- `expo-notifications` `~0.32.17`
- `expo-router` `~6.0.24`
- `expo-secure-store` `~15.0.8`
- `@types/react` `~19.1.17`
- `@react-navigation/bottom-tabs` declared as `^7.4.0`

`expo-secure-store` is also registered as an Expo config plugin because the SDK 54-compatible package exposes native configuration through the plugin system.

The root `pnpm-lock.yaml` is the canonical frozen resolution for this manifest.

## Cross-workspace React type authority

The first full PR matrix exposed a second-order monorepo regression after the native alignment: mobile moved to `@types/react ~19.1.17`, while Web still pinned `@types/react 19.1.9`. The shared lockfile then contained two React type universes, and Web `tsc` failed with incompatible `Ref` definitions and DOM attribute differences such as `popover="hint"`.

The repair does not modify runtime React or UI implementation code. Web now declares the same direct `@types/react` authority as mobile, `~19.1.17`, and the canonical lockfile is regenerated from that unified manifest state.

The repair was materialized only after one combined gate proved:

- Web type-check succeeds;
- `expo install --check --pnpm` remains clean;
- the mobile client type-checks; and
- native lint remains zero-warning.

## Permanent CI authority

`.github/workflows/expo-sdk-compatibility-ci.yml` is read-only. It:

1. installs the frozen workspace dependency graph;
2. runs `expo install --check --pnpm` inside the mobile app;
3. requires Web and mobile to declare the same direct `@types/react` range;
4. type-checks the native client;
5. type-checks the Web client against the same frozen lockfile;
6. lints the native client with the repository's zero-warning policy; and
7. fails if any of those checks mutate the checkout.

The lane triggers on both mobile and Web package-manifest changes, as well as the root lockfile, so a future SDK update cannot silently split the monorepo's React type authority again.

## Relationship to iOS release qualification

This compatibility tranche does not by itself prove a signed, archived, TestFlight-ready, or App Store-ready binary.

After this graph is merged, the iOS CocoaPods/Xcode qualification tranche must be restacked and rerun. That gate owns:

- actual CocoaPods resolution and `Podfile.lock` evidence;
- the final linked native pod/resource graph;
- rejection of unexpected Google Maps iOS authority;
- Xcode 26 Release simulator compilation with signing disabled; and
- inspection of the built `.app` bundle, including the bundled privacy manifest.

The generated-native privacy audit must also rerun because changing native package versions can change required-reason API declarations.

## Explicit non-claims

This tranche does not claim:

- Apple signing or provisioning qualification;
- EAS project/account qualification;
- TestFlight installation;
- physical-device behavior;
- App Store review readiness; or
- production deployment availability.

Those remain separate release authorities and evidence gates.
