# iOS Physical Device + Xcode v1

## Goal

This path is for viewing the current Woof native application on a real iPhone from Xcode with the
same generated native graph that repository CI qualifies.

The generated `apps/mobile/ios` directory is an **artifact**, not source authority. Expo config,
JavaScript/TypeScript source, the lockfile, and CocoaPods resolution remain the reproducible inputs.

## Fastest first launch

From the repository root on the Mac that has Xcode installed:

```bash
pnpm --filter @woof/mobile ios:device:prepare
```

The helper defaults to the repository-qualified remote API:

```text
https://woof-api-prod.fly.dev/api/v1
```

That is the lowest-friction route for inspecting the current iPhone product because the phone does
not depend on a development server running on the Mac.

The helper:

1. verifies macOS, Xcode, CocoaPods, Node, pnpm and curl;
2. resolves the selected API authority;
3. chooses a development-only bundle identity for local Apple signing;
4. writes ignored `apps/mobile/.env.local` device authority;
5. checks the resolved Expo public config;
6. runs a clean iOS Expo prebuild;
7. resolves CocoaPods;
8. writes ignored `ios/.xcode.env.local` with the active Node/config authority;
9. verifies `Woof.xcworkspace` exists;
10. opens the workspace.

## Xcode steps

After preparation:

1. connect the iPhone to the Mac and unlock it;
2. trust the Mac if iOS asks;
3. enable **Developer Mode** on the iPhone if required;
4. open the **Woof** target in Xcode;
5. open **Signing & Capabilities**;
6. enable automatic signing and choose your Apple Development team;
7. select the physical iPhone in the run-destination menu;
8. start Metro in another terminal;
9. press **Cmd+R**.

Metro command:

```bash
pnpm --filter @woof/mobile ios:device:metro
```

The preparation command normally opens the correct workspace automatically. To reopen it later:

```bash
pnpm --filter @woof/mobile ios:xcode
```

Always open `Woof.xcworkspace`, not the nested `.xcodeproj`, because CocoaPods owns native
dependencies through the workspace.

## Development bundle identity

Production remains `com.woof.app`.

Physical development builds may override that identity through:

```text
WOOF_IOS_BUNDLE_IDENTIFIER
```

The helper defaults to:

```text
com.sidhulyalkar.woof.dev
```

This prevents a local signing test from requiring possession of the production App ID. If Apple
reports that the development identifier is unavailable, choose another reverse-DNS identifier:

```bash
pnpm --filter @woof/mobile ios:device:prepare -- \
  --bundle-id com.<your-unique-name>.woof.dev
```

Preview and production config reject a development-only override that differs from the canonical
release identifier.

## Exact local API mode

To test the checked-out API rather than the qualified remote environment, first start the repository
dependencies/API, then prepare with:

```bash
pnpm --filter @woof/mobile ios:device:prepare -- --local-api
```

The helper resolves the active Mac network interface and writes a URL such as:

```text
http://192.168.1.42:4000/api/v1
```

It deliberately rejects `localhost`, because on an iPhone that address refers to the phone.

For development-only cleartext LAN mode, dynamic Expo config adds local-network permission text and
development-only App Transport Security allowances. Those allowances are not added to preview or
production config.

If the API health check is unreachable:

- confirm the Mac and iPhone are on the same network;
- confirm the API is listening on port 4000;
- allow incoming Node connections in the macOS firewall if prompted;
- verify `http://<mac-ip>:4000/api/v1/health` from the Mac first.

## Debug vs standalone behavior

For normal iteration, use **Debug + Metro**. JavaScript changes refresh quickly without rebuilding
the complete native graph.

A Release device build bundles JavaScript and does not use the same rapid Metro loop, but signing and
release configuration are separate authorities. The initial physical qualification should remain
Debug until the full authenticated journey is proven.

## What you should test on the first iPhone pass

Use a real account and walk the canonical native spine:

1. launch cold;
2. register or sign in;
3. resolve Guardian/Companion mode;
4. complete pet setup if needed;
5. load Today;
6. start and finish an Adventure;
7. record dog and owner outcome separately;
8. open Compass;
9. open Story;
10. inspect Community;
11. open Daily Signals, Pets, Goals, Library, Events, Map and Profile;
12. background and foreground the app;
13. terminate and cold-launch again;
14. verify logout and account/session recovery behavior.

Record any screen that clips, stalls, uses stale/demo data, loses authentication, or fails on a real
network. That is the evidence needed for the next native polish tranche.

## Evidence boundary

Repository CI already proves generated iOS composition, CocoaPods resolution, unsigned Xcode
Simulator compilation, bundle/privacy authority and controlled Simulator launch.

This helper does not itself prove:

- physical iPhone execution;
- Apple signing/provisioning ownership;
- production/TestFlight archive validity;
- APNs delivery;
- complete native social parity;
- App Store readiness.

The first successful real-device journey is new evidence and should be recorded against the exact Git
SHA, Xcode version, iOS version, device model, API authority and bundle identifier used.
