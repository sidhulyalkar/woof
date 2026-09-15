#!/usr/bin/env python3
"""Fail closed when the first native accessibility reality boundary drifts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAVIGATOR = ROOT / "apps/mobile/src/navigation/AppNavigator.tsx"
REDUCED_MOTION = ROOT / "apps/mobile/src/accessibility/useReducedMotionPreference.ts"
PRIMARY_SCREENS = [
    ROOT / "apps/mobile/src/screens/TodayScreen.tsx",
    ROOT / "apps/mobile/src/screens/FirstAdventureScreen.tsx",
    ROOT / "apps/mobile/src/screens/DailySignalsScreen.tsx",
]

for path in [NAVIGATOR, REDUCED_MOTION, *PRIMARY_SCREENS]:
    if not path.is_file():
        raise SystemExit(f"native accessibility authority source missing: {path.relative_to(ROOT)}")

navigator = NAVIGATOR.read_text()
reduced_motion = REDUCED_MOTION.read_text()

required_navigator_markers = [
    "useReducedMotionPreference",
    "animation: reduceMotionEnabled ? 'none' : 'default'",
    "KeyboardAvoidingView",
    "keyboardAvoidanceBehavior",
    "DailySignalsKeyboardSafeScreen",
    '<View style={styles.loadingContainer} accessibilityRole="progressbar">',
]
for marker in required_navigator_markers:
    if marker not in navigator:
        raise SystemExit(f"native navigation accessibility marker missing: {marker}")

if "height: 66" in navigator:
    raise SystemExit("native tab bar must remain content/safe-area driven, not fixed to 66px")

required_motion_markers = [
    "AccessibilityInfo.isReduceMotionEnabled()",
    "AccessibilityInfo.addEventListener(",
    "'reduceMotionChanged'",
    "subscription.remove()",
]
for marker in required_motion_markers:
    if marker not in reduced_motion:
        raise SystemExit(f"reduced-motion authority marker missing: {marker}")

for path in (ROOT / "apps/mobile/src").rglob("*.tsx"):
    text = path.read_text()
    for forbidden in ["allowFontScaling={false}", "maxFontSizeMultiplier={1}"]:
        if forbidden in text:
            raise SystemExit(
                f"native source disables user text scaling in {path.relative_to(ROOT)}: {forbidden}"
            )

print(
    "Native accessibility foundation is explicit: tab navigation is content-driven, stack motion follows "
    "the OS reduced-motion preference, launch/capture loading is announced, keyboard-sensitive capture "
    "surfaces are protected, and text scaling remains user-controlled. Physical-device VoiceOver and "
    "Dynamic Type usability remain separate evidence gates."
)
