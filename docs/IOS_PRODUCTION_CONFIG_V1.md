# iOS Production Configuration v1

## Purpose

This tranche removes configuration states that could produce a signed Woof binary pointed at localhost or linked to a placeholder Expo Application Services (EAS) project.

It is a release-configuration boundary, not evidence that an App Store or TestFlight build exists.

## Build authority

Woof now resolves native build configuration through `apps/mobile/app.config.ts`.

The static `app.json` remains development-friendly and may contain the local API default used by Metro during local iteration. It no longer contains a fake EAS project id.

The dynamic config derives the build profile from `EAS_BUILD_PROFILE`, with `WOOF_BUILD_PROFILE` available only as a deterministic local/CI qualification input.

### Development

Development may use:

- `http://localhost:4000/api/v1`;
- no EAS project id for ordinary local Expo iteration.

This preserves a low-friction local development loop.

### Preview and production

Any non-development build fails before native generation unless:

- `EXPO_PUBLIC_API_URL` is explicitly supplied;
- the API URL is an absolute remote HTTPS URL;
- the API host is not localhost or another loopback address;
- a real EAS project id is supplied through `EAS_PROJECT_ID`, `EAS_BUILD_PROJECT_ID`, or linked static config.

The runtime Axios client repeats the remote-HTTPS guard so a non-development binary cannot silently fall back to localhost even if an upstream configuration step regresses.

## EAS environments

`apps/mobile/eas.json` binds each build profile to the matching EAS environment:

- `development -> development`;
- `preview -> preview`;
- `production -> production`.

The production environment must own the public `EXPO_PUBLIC_API_URL` value used by the release build.

Client-side `EXPO_PUBLIC_*` values are public by design and must never contain credentials or secrets.

## Version authority

EAS developer-facing versions now use the remote version source. Production builds enable `autoIncrement` so repeated TestFlight/App Store archives cannot accidentally reuse a build number because a local source file was not bumped.

The user-facing app version remains `1.0.0` until product release policy intentionally changes it.

## Current production API authority

The repository production deployment workflow currently qualifies:

`https://woof-api-prod.fly.dev/api/v1`

The iOS config qualification lane uses that URL as its known-good production scenario. The dynamic config still requires the selected EAS environment to provide the value so API authority is explicit at build time rather than hidden in mobile source.

If the canonical production API moves, deployment and native configuration evidence must move together.

## Qualification

`iOS Production Config CI` proves all of the following from a frozen dependency install:

1. source-level production configuration invariants;
2. canonical formatting;
3. development config still resolves with the local API fallback;
4. production config rejects a missing API URL;
5. production config rejects a loopback/non-HTTPS API URL;
6. production config rejects a missing EAS project id;
7. production config resolves with a qualified remote API URL plus test project id;
8. the full native client type-checks and lints with zero warnings.

The test project id used in CI is intentionally synthetic. It proves configuration behavior only and is not EAS project authority.

## Required owner/EAS action before a real build

Repository code cannot manufacture a legitimate Expo project identity. Before the first real preview or production EAS build, Woof still needs:

1. the actual Expo/EAS project to exist;
2. its real project id to be linked/supplied as `EAS_PROJECT_ID` or canonical app config;
3. the production EAS environment to contain the canonical `EXPO_PUBLIC_API_URL`;
4. iOS signing credentials/provisioning to be established through the authorized Apple Developer account;
5. remote iOS build-number authority to be initialized if an earlier App Store/TestFlight build number already exists.

Those are provider/account authorities and must not be replaced with repository placeholders.

## Explicit non-claims

This release does **not** claim:

- possession of an actual EAS project id;
- Apple signing credentials;
- a successful EAS cloud build;
- an `.ipa` archive;
- TestFlight upload or installation;
- physical iPhone qualification;
- App Store Connect metadata completion;
- privacy-manifest inspection of a generated iOS archive;
- APNs qualification;
- App Store approval.

## Exit condition

This tranche is complete when the repository can truthfully say:

> Development may intentionally use localhost, but preview and production Woof builds cannot resolve unless they are bound to explicit remote API and EAS project authority.
