# iOS Simulator Accessibility Reality v1

Issue: #155

## Purpose

The CocoaPods/Xcode lane already proves that Woof can generate, resolve, and compile a production-shaped unsigned iOS Simulator application. This extension takes one additional step: it installs and launches that exact audited `Woof.app` on a real iPhone Simulator under controlled Dynamic Type settings.

This is executable simulator evidence. It is still not VoiceOver journey evidence, TestFlight evidence, or physical-device evidence.

## Chain of custody

The simulator steps do not independently search for an arbitrary app bundle. `audit-ios-cocoapods-build.py` first requires exactly one simulator `Woof.app`, verifies its `com.woof.app` bundle identity and privacy manifest, and records its repository-relative path in `ios-cocoapods-build-report.json`.

The runtime steps consume that audited path, verify the bundle identifier again, install it into the selected simulator, and record the resulting app container.

The evidence chain is therefore:

> production-shaped prebuild -> resolved Pods -> Release simulator build -> audited Woof.app -> installed Woof.app -> launched Woof.app

## Simulator authority

The workflow runs on the same `macos-26` / Xcode 26 job as native compilation. It inventories all available CoreSimulator devices and chooses an available iPhone from the newest installed iOS runtime. The exact name, UDID, and runtime identifier are retained in `ios-simulator-selection.json`.

Before use, the selected simulator is shut down if necessary, erased, booted, and waited to the boot-complete boundary. Erasing the hosted-runner simulator prevents stale app state, credentials, or prior test data from becoming part of the launch result.

No test account is provisioned in this tranche. A fresh Woof install therefore exercises the unauthenticated app shell only.

## Dynamic Type authority

Before attempting to set text size, the workflow records `xcrun simctl help ui` from the actual Xcode 26 runner and requires that the installed tool advertises `content_size`. This makes the runner's own CLI vocabulary part of the evidence rather than assuming an older Xcode command contract.

The lane then:

1. sets `content_size large` as an explicit baseline;
2. records the simulator's reported baseline content size;
3. launches Woof, waits five seconds, captures a screenshot, and requires the app to remain terminable;
4. sets `content_size accessibility-extra-extra-extra-large`;
5. records the simulator's reported accessibility content size and requires it to differ from the baseline report;
6. launches the same installed Woof bundle again, waits five seconds, captures a second screenshot, and again requires the app to remain terminable;
7. records SHA-256 hashes of both screenshots for retained evidence.

`accessibility-extra-extra-extra-large` corresponds to UIKit's largest accessibility content-size category. The workflow does not infer visual correctness from the screenshot hashes; the images are retained for inspection.

## What a green lane proves

A green exact-head lane proves that, on the recorded GitHub-hosted macOS/Xcode toolchain:

- the production-shaped native project still resolves and compiles;
- the audited unsigned Release simulator app can be installed into a clean available iPhone Simulator;
- the runner supports the Dynamic Type control used by the lane;
- the simulator reports a changed content-size setting between baseline and the largest accessibility category;
- `com.woof.app` launches at both settings;
- the launched app survives the five-second observation window long enough to capture a non-empty screenshot and accept an explicit terminate command.

This is materially stronger than build-only evidence, but deliberately narrower than an accessibility usability claim.

## What it does not prove

A green lane does **not** prove:

- that the authenticated First Adventure -> Today -> Daily Signals -> Adventure outcome -> Story journey works at the largest Dynamic Type size;
- that every screen reflows without clipping, truncation, overlap, or inaccessible off-screen controls;
- VoiceOver reading order, focus order, labels, hints, rotor behavior, or action announcements;
- that React Native observed a reduced-motion preference during this automated launch;
- keyboard behavior on a particular simulated or physical device;
- a signed EAS/TestFlight build;
- a physical iPhone launch;
- App Store readiness;
- live production API availability;
- pilot validation.

The fresh simulator has no Woof credentials, so this tranche intentionally does not mutate user or production data.

## Retained evidence

The existing 14-day native artifact now also retains:

- the available simulator inventory;
- exact selected simulator metadata;
- boot/boot-status output;
- `simctl ui` help output;
- installed Woof app-container path;
- baseline and largest reported content-size values;
- baseline and largest launch output;
- baseline and largest screenshots;
- screenshot SHA-256 hashes.

## Next evidence step

After this launch boundary is qualified, #155 still needs a way to enter the maintained authenticated relationship journey without embedding credentials or bypassing server authority. The preferred path is a disposable staging account against a real staging API once staging is live, followed by either a minimal XCUITest journey or controlled manual simulator execution with retained evidence.

VoiceOver should remain a manual simulator/physical-device gate until Woof has a trustworthy automation path for assistive-technology focus and action behavior.

Keep the evidence states separate:

> source-qualified != simulator-launch-qualified != authenticated-journey-qualified != physical-device-qualified != pilot-validated
