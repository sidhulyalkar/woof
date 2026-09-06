# iOS Native Artifact Audit v1

## Purpose

Woof uses Expo Continuous Native Generation. Repository JavaScript and app-config correctness therefore is not the final iOS authority: App Store-facing metadata is produced when Expo prebuild materializes the native Xcode project.

This tranche creates an evidence-only iOS prebuild gate before signing or TestFlight.

## Production-shaped generation

CI evaluates the same fail-closed production configuration contract as the release build with:

- production build profile semantics;
- the canonical public production API URL;
- a synthetic EAS project UUID used only to let configuration resolve in CI.

It then runs Expo prebuild for iOS with `--clean --no-install`.

This generates the Xcode project and app metadata without requiring CocoaPods, an Apple Developer account, signing certificates, provisioning profiles, or a real EAS project.

## Generated metadata authority

The audit requires the generated native project to preserve:

- bundle identifier `com.woof.app`;
- camera usage description;
- selected-photo-library usage description;
- location-while-in-use description.

`app.json` is also required to keep the `ios.infoPlist` permission strings identical to the corresponding Expo plugin permission options. This prevents two source authorities from silently disagreeing while prebuild lets the plugin override win.

The test reads the generated plist/project files rather than merely rereading `app.json`. Metadata discrepancies are collected into the report instead of aborting evidence collection at the first mismatch.

## Native-linkage-aware privacy inventory

Apple privacy-manifest correctness cannot be inferred from every file present under `node_modules`.

The audit therefore retains both Expo Apple autolinking and React Native iOS autolinking resolver output, derives the candidate native package roots from those resolvers, reads the generated Podfile, and inventories `PrivacyInfo.xcprivacy` only from that pre-Pods candidate graph.

For `react-native-maps`, the generated Podfile does not enable the separate Google Maps subspec, so the `AirGoogleMaps` privacy bundle is explicitly excluded from the candidate union. This prevents an optional native implementation that Woof does not configure from broadening the app declaration.

The machine-readable report and resolver evidence are retained as GitHub Actions artifacts for 14 days even when the audit fails.

## Evidence-backed required reasons

The linkage-aware candidate graph declares these required-reason APIs:

- `NSPrivacyAccessedAPICategoryDiskSpace`: `85F4.1`, `E174.1`
- `NSPrivacyAccessedAPICategoryFileTimestamp`: `0A2A.1`, `3B52.1`, `C617.1`
- `NSPrivacyAccessedAPICategorySystemBootTime`: `35F9.1`
- `NSPrivacyAccessedAPICategoryUserDefaults`: `CA92.1`

Woof repeats exactly that candidate set through `expo.ios.privacyManifests` because Expo documents that Apple does not correctly parse every privacy manifest shipped through static CocoaPods dependencies and may require dependency reasons to be repeated at app level.

A production-shaped prebuild now materializes one app-target `apps/mobile/ios/Woof/PrivacyInfo.xcprivacy` containing those categories and reasons. The generated manifest also declares `NSPrivacyTracking=false`, no tracking domains, and no app-level collected-data types.

The successful generated-native evidence artifact for the evidence-backed manifest had digest:

`sha256:3a3eafec1e539be962d5cf375c8a75b9bc4d81d43b67f683ab0203bda2239f98`

## Findings resolved during this tranche

### Permission source split

The first generated prebuild showed two different source copies for the same iOS permission descriptions: older strings under `ios.infoPlist` and newer strings in the Expo camera, image-picker and location plugin options. Expo prebuild materialized the plugin strings.

The source values now converge on the generated wording.

### Raw dependency overcount

A raw scan of installed packages found multiple transitive Expo module versions and the optional `react-native-maps` Google privacy bundle. The native-linkage-aware resolver reduced this to the actual pre-Pods candidate graph and removed the unsupported Google Maps `1C8F.1` UserDefaults reason from Woof's app-level declaration.

## Why this is stricter than source inspection

Expo config plugins modify `Info.plist`, Xcode project settings, privacy manifests, entitlements and related files during prebuild. A source-only assertion can therefore pass while generated native state differs.

This lane moves Woof's release evidence one layer closer to what Apple actually receives.

## Remaining authority boundary

This is still a **pre-Pods** audit. Autolinking plus the generated Podfile is much stronger than scanning the install graph, but it is not equivalent to a resolved `Podfile.lock` or a compiled app bundle.

The next native tranche must run on macOS with CocoaPods and Xcode, inspect the actually resolved pods and privacy resources, and build the generated project without claiming signing. Any difference between that resolved native graph and this candidate set must update the app manifest before TestFlight.

The dependency manifests may also contain collected-data declarations that Apple/Xcode merges independently. This tranche does not copy those declarations into Woof's app manifest merely because they exist in a package; the resolved CocoaPods/archive gate should determine the final merged evidence.

## Explicit non-claims

Passing this audit does not claim:

- CocoaPods installation or `Podfile.lock` authority;
- final static-pod manifest aggregation;
- a compiled `.app` or `.ipa`;
- Xcode 26 compilation;
- code signing or provisioning;
- a real Expo/EAS project id;
- TestFlight upload;
- Apple's server-side required-reason validation;
- App Store Connect privacy-label completion;
- physical-device behavior.

Those remain later release gates.

## Exit condition

This tranche is complete when Woof can truthfully say:

> Production-shaped Expo prebuild deterministically produces the expected iOS bundle and permission metadata, generates an app-level privacy manifest covering the linkage-aware required-reason candidate set, and retains machine-readable native-linkage evidence for the next CocoaPods/Xcode qualification stage.
