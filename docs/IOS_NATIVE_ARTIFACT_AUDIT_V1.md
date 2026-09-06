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

## Privacy manifest inventory

Apple privacy-manifest correctness cannot be inferred from package names alone.

The audit inventories:

- app-target `PrivacyInfo.xcprivacy` files generated under `apps/mobile/ios`;
- parseable `PrivacyInfo.xcprivacy` files shipped by installed dependency packages;
- the union of dependency `NSPrivacyAccessedAPITypes` and their declared required reasons;
- manifest parse failures, which fail the lane rather than being ignored.

The machine-readable report is retained as a GitHub Actions artifact for 14 days even when the audit fails.

The audit intentionally **does not invent or broaden required-reason declarations**. A missing generated app privacy manifest is only a hard failure when the installed dependency inventory declares required-reason API categories. In that case the report becomes the evidence for the next minimal repair.

This is intentionally conservative for Expo projects because Expo documents that Apple does not correctly parse every `PrivacyInfo.xcprivacy` shipped through static CocoaPods dependencies and may require those dependency reasons to be repeated in the app-level privacy manifest.

## First audit finding

The first generated prebuild showed that Woof had two different source copies for the same iOS permission descriptions: older strings under `ios.infoPlist` and newer strings in the Expo camera, image-picker and location plugin options. Expo prebuild materialized the plugin strings.

The tranche now converges those source values before continuing privacy-manifest analysis.

## Why this is stricter than source inspection

Expo config plugins modify `Info.plist`, Xcode project settings, privacy manifests, entitlements and related files during prebuild. A source-only assertion can therefore pass while generated native state differs.

This lane moves Woof's release evidence one layer closer to what Apple actually receives.

## Explicit non-claims

Passing this audit does not claim:

- CocoaPods installation or static-pod manifest aggregation;
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

> Production-shaped Expo prebuild deterministically produces the expected iOS bundle/permission metadata, and Woof has a machine-readable inventory of the privacy manifests and required-reason declarations present in its generated app and installed native dependencies.
