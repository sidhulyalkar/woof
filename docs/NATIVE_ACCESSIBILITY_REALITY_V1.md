# Native Accessibility Reality v1

Issue: #155

## Purpose

This work establishes a repository-qualified accessibility foundation for the maintained React Native client without pretending that source code or an unsigned simulator build proves physical-device usability.

The boundary now covers platform behavior plus the first primary-screen reflow and control-semantics tranche:

- navigation must not force a fixed tab-bar height that can clip scaled labels or fight safe-area insets;
- stack transitions must respect the operating system reduced-motion preference;
- keyboard-sensitive First Adventure and Daily Signals capture surfaces must remain visible while text input is active;
- launch and authority-loading states must expose progress semantics and readable status text;
- user font scaling must remain enabled rather than being capped to preserve a default-size layout;
- Today relationship-tool cards must wrap from a flexible basis instead of a fixed percentage width;
- First Adventure alternate-role actions must wrap instead of being forced into one horizontal pair;
- the icon-only Today outcome-close action must expose a stable accessible name and at least a 44-point target.

## Reduced motion

`useReducedMotionPreference` reads `AccessibilityInfo.isReduceMotionEnabled()` and subscribes to `reduceMotionChanged` for the lifetime of the mounted navigator.

Auth, Guardian, and Companion stacks select React Navigation's `animation: 'none'` when reduced motion is enabled and `animation: 'default'` otherwise. The product therefore responds to preference changes while running instead of sampling the preference only once at install or startup.

The hook fails open to normal navigation if the platform cannot resolve the preference. Accessibility preference lookup must not strand the user outside the app.

## Dynamic Type and reflow boundary

The foundation removes the explicit `height: 66` bottom-tab constraint. Tab content and platform safe-area behavior can determine the resulting height.

The follow-up reflow tranche removes Today's fixed `width: '48%'` relationship-tool cards. Tool cards now use `flexGrow` plus a 220-point `flexBasis`, allowing narrow mobile layouts to stack naturally while wider layouts can still form columns. Card height remains content-driven, so scaled labels and captions can add vertical space instead of being clipped into a fixed box.

First Adventure's alternate "learn" and "foster / support" actions likewise use a wrap-capable row plus the same flexible basis rather than two equal horizontal columns. This preserves the existing choices and mode authority while giving the layout room to stack when horizontal space is constrained.

The source contract also fails if maintained React Native TSX disables text scaling with `allowFontScaling={false}` or hard-caps text with `maxFontSizeMultiplier={1}`.

These source/layout guarantees reduce known compression risks. They do **not** prove every primary screen is usable at the largest accessibility Dynamic Type size. Largest-text simulator and physical-device execution remain explicit evidence gates.

## Control semantics

Today relationship-tool buttons expose a concise combined accessible label using each tool's visible name and caption.

The outcome sheet's icon-only close control exposes `Close outcome check-in` as its accessible name and uses a minimum 44 by 44 point target. The visual icon remains decorative inside that named parent action.

This does not establish complete VoiceOver focus order or announcement quality. Those require running the built app with VoiceOver enabled.

## Keyboard boundary

First Adventure pet setup and Daily Signals capture both contain meaningful text entry. They are wrapped in a `KeyboardAvoidingView` at the maintained navigation boundary. iOS uses `padding` behavior; other platforms keep their normal resize behavior.

The screen-owned `ScrollView` remains responsible for scrolling and tap persistence. The navigator does not duplicate form state or validation.

This is a structural keyboard-avoidance guarantee, not proof of every keyboard/device combination. Simulator and physical-device checks remain required.

## Qualification

`Native Accessibility Reality CI` proves:

- required reduced-motion and keyboard-avoidance source markers remain present;
- the tab bar does not return to the retired fixed 66-point height;
- Today relationship tools cannot return to the fixed 48-percent card width;
- First Adventure alternate actions remain wrap-capable;
- the named Today close control and 44-point target remain present;
- maintained mobile TSX does not disable user font scaling in the simple forbidden forms covered by the sentry;
- the dedicated files are formatted;
- the complete mobile package lints and type-checks.

Any `apps/mobile/**` change also triggers the existing iOS CocoaPods + Xcode Qualification CI, which generates a production-shaped iOS project, resolves Pods, and compiles an unsigned Release simulator application with Xcode 26.

## Evidence boundary

Repository/source qualification can establish intentional semantics and compile-time compatibility. The Xcode lane can establish that a production-shaped simulator application builds. Neither proves VoiceOver reading/focus order, largest-Dynamic-Type usability, keyboard presentation on a particular device, or TestFlight behavior.

Keep the evidence states separate:

> source-qualified != simulator-qualified != physical-device-qualified != pilot-validated

The next #155 evidence step should execute the maintained Guardian and First Adventure surfaces with largest accessibility text and VoiceOver in a simulator, then repeat the launch-critical path on a physical iPhone/TestFlight build before making device-usability claims.
