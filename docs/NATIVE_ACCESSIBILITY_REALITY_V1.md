# Native Accessibility Reality v1

Issue: #155

## Purpose

This tranche establishes a repository-qualified accessibility foundation for the maintained React Native client without pretending that source code or an unsigned simulator build proves physical-device usability.

The first boundary focuses on platform behavior that can be made explicit without redesigning the product:

- navigation must not force a fixed tab-bar height that can clip scaled labels or fight safe-area insets;
- stack transitions must respect the operating system reduced-motion preference;
- keyboard-sensitive First Adventure and Daily Signals capture surfaces must remain visible while text input is active;
- launch and authority-loading states must expose progress semantics and readable status text;
- user font scaling must remain enabled rather than being capped to preserve a default-size layout.

## Reduced motion

`useReducedMotionPreference` reads `AccessibilityInfo.isReduceMotionEnabled()` and subscribes to `reduceMotionChanged` for the lifetime of the mounted navigator.

Auth, Guardian, and Companion stacks select React Navigation's `animation: 'none'` when reduced motion is enabled and `animation: 'default'` otherwise. The product therefore responds to preference changes while running instead of sampling the preference only once at install or startup.

The hook fails open to normal navigation if the platform cannot resolve the preference. Accessibility preference lookup must not strand the user outside the app.

## Dynamic Type and reflow boundary

This release removes the explicit `height: 66` bottom-tab constraint. Tab content and platform safe-area behavior can determine the resulting height.

The source contract also fails if maintained React Native TSX disables text scaling with `allowFontScaling={false}` or hard-caps text with `maxFontSizeMultiplier={1}`.

This does **not** yet prove that every primary screen reflows correctly at the largest accessibility text sizes. Today relationship-tool card reflow and First Adventure alternative-action reflow remain explicit follow-up work in #155 and must be exercised with large Dynamic Type before stronger claims are made.

## Keyboard boundary

First Adventure pet setup and Daily Signals capture both contain meaningful text entry. They are wrapped in a `KeyboardAvoidingView` at the maintained navigation boundary. iOS uses `padding` behavior; other platforms keep their normal resize behavior.

The screen-owned `ScrollView` remains responsible for scrolling and tap persistence. The navigator does not duplicate form state or validation.

This is a structural keyboard-avoidance guarantee, not proof of every keyboard/device combination. Simulator and physical-device checks remain required.

## Qualification

`Native Accessibility Reality CI` proves:

- required reduced-motion and keyboard-avoidance source markers remain present;
- the tab bar does not return to the retired fixed 66-point height;
- maintained mobile TSX does not disable user font scaling in the simple forbidden forms covered by the sentry;
- the dedicated files are formatted;
- the complete mobile package lints and type-checks.

Any `apps/mobile/**` change also triggers the existing iOS CocoaPods + Xcode Qualification CI, which generates a production-shaped iOS project, resolves Pods, and compiles an unsigned Release simulator application with Xcode 26.

## Evidence boundary

Repository/source qualification can establish intentional semantics and compile-time compatibility. The Xcode lane can establish that a production-shaped simulator application builds. Neither proves VoiceOver reading/focus order, largest-Dynamic-Type usability, keyboard presentation on a particular device, or TestFlight behavior.

Keep the evidence states separate:

> source-qualified != simulator-qualified != physical-device-qualified != pilot-validated

The next #155 tranche should target actual large-text reflow and explicit control labels/states in Today, First Adventure, and Daily Signals, followed by simulator and physical-iPhone execution evidence.
