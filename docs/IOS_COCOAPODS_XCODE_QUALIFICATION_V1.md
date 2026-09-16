# iOS CocoaPods + Xcode Qualification v1

## Purpose

The prebuild/native-artifact audit proves what Expo intends to generate before CocoaPods resolves the native graph. This tranche moves closer to the binary Apple receives by resolving Pods on macOS, compiling the generated iOS workspace with Xcode 26, auditing the built bundle, and then installing and launching that same audited bundle on a clean iPhone Simulator.

It remains intentionally unsigned. Signing, provisioning, EAS credentials, TestFlight and physical-device behavior are separate authorities.

The simulator accessibility extension is documented in `IOS_SIMULATOR_ACCESSIBILITY_REALITY_V1.md`.

## Environment authority

The qualification lane runs on GitHub's `macos-26` hosted image and records:

- macOS version;
- exact `xcodebuild -version` output;
- selected developer directory;
- CocoaPods version.

The lane fails if the selected Xcode major version is not 26.

## Native composition sequence

The lane:

1. installs the repository from the frozen pnpm lockfile;
2. runs production-shaped Expo iOS prebuild with the qualified API/config contract;
3. runs CocoaPods resolution and requires a generated `Podfile.lock` and `Woof.xcworkspace`;
4. inventories the resolved workspace;
5. compiles the `Woof` Release configuration for the generic iOS Simulator with code signing disabled;
6. inspects the built `.app` rather than only source/generated project files;
7. selects and erases an available iPhone Simulator from the newest installed iOS runtime;
8. verifies the runner's own `simctl ui` command exposes Dynamic Type control;
9. installs the exact audited `Woof.app`;
10. launches it at an explicit baseline content size and at the largest accessibility content size, retaining launch and screenshot evidence.

## CocoaPods authority

`Podfile.lock` becomes the resolved pre-binary dependency authority for this tranche.

Woof does not currently configure the optional Google Maps implementation in `react-native-maps`, so the resolved lock must not unexpectedly contain Google Maps pod authority. If it does, the lane fails and the privacy/config boundary must be revisited.

The audit also records privacy-manifest file references present in the resolved Pods Xcode project.

## Built app privacy authority

The compiled simulator app must contain at its bundle root:

- `Info.plist` with bundle identifier `com.woof.app`;
- `PrivacyInfo.xcprivacy`.

The built app's required-reason union must exactly equal the `expo.ios.privacyManifests.NSPrivacyAccessedAPITypes` authority committed in `app.json`.

This is stronger than merely seeing a generated privacy manifest under the source `ios` directory because Xcode must actually copy it into the built product.

## Simulator launch authority

The bundle audit records the one qualified simulator `Woof.app` path in `ios-cocoapods-build-report.json`. The simulator stage consumes that exact path, verifies `com.woof.app` again, installs it into an erased iPhone Simulator, and records the installed app container.

The lane explicitly sets Dynamic Type to `large`, launches and observes the app, then sets the simulator to `accessibility-extra-extra-extra-large` and repeats the launch. The simulator's reported content-size values must differ. Both launches must remain alive through the five-second observation window long enough to capture a screenshot and accept an explicit terminate command.

This establishes launch reality under a controlled accessibility setting. It does not establish screen-by-screen reflow quality or an authenticated user journey.

## Retained evidence

The workflow retains for 14 days:

- Apple toolchain identity;
- CocoaPods install log;
- generated Podfile and resolved Podfile.lock;
- resolved Pods project;
- workspace inventory;
- complete Xcode build log;
- machine-readable CocoaPods/build report;
- built app Info.plist and PrivacyInfo.xcprivacy;
- available simulator inventory and selected simulator identity;
- simulator boot status;
- `simctl ui` help output from the actual runner;
- installed app-container path;
- baseline and largest accessibility content-size reports;
- both launch outputs;
- both simulator screenshots and their SHA-256 hashes.

Evidence is uploaded even when qualification fails so a failing native composition or simulator launch can be inspected rather than retried blindly.

## Explicit non-claims

Passing this lane does **not** claim:

- an App Store-signed build;
- distribution provisioning profiles;
- a real EAS project identity;
- a production `.ipa`;
- Apple App Store Connect server-side privacy validation;
- TestFlight upload or external review;
- APNs production behavior;
- an authenticated Guardian/Companion journey in Simulator;
- complete largest-Dynamic-Type reflow usability;
- VoiceOver focus/order/action behavior;
- reduced-motion runtime behavior;
- physical iPhone behavior;
- that live production currently runs the same SHA.

The separate production deployment blocker remains tracked in issue #77. Native accessibility execution remains tracked in issue #155.

## Exit condition

This tranche is complete when Woof can truthfully say:

> On a recorded Xcode 26/macOS 26 toolchain, CocoaPods resolves the production-shaped native project, the unsigned Release iOS Simulator app compiles with the expected bundle/privacy authority, that exact audited bundle installs into a clean iPhone Simulator, and Woof remains launchable at both baseline and the largest accessibility Dynamic Type settings.
