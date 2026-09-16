#!/usr/bin/env python3
"""Fail closed when the native accessibility reality boundary drifts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAVIGATOR = ROOT / "apps/mobile/src/navigation/AppNavigator.tsx"
REDUCED_MOTION = ROOT / "apps/mobile/src/accessibility/useReducedMotionPreference.ts"
TODAY = ROOT / "apps/mobile/src/screens/TodayScreen.tsx"
FIRST_ADVENTURE = ROOT / "apps/mobile/src/screens/FirstAdventureScreen.tsx"
DAILY_SIGNALS = ROOT / "apps/mobile/src/screens/DailySignalsScreen.tsx"
PRIMARY_SCREENS = [TODAY, FIRST_ADVENTURE, DAILY_SIGNALS]

for path in [NAVIGATOR, REDUCED_MOTION, *PRIMARY_SCREENS]:
    if not path.is_file():
        raise SystemExit(f"native accessibility authority source missing: {path.relative_to(ROOT)}")

navigator = NAVIGATOR.read_text()
reduced_motion = REDUCED_MOTION.read_text()
today = TODAY.read_text()
first_adventure = FIRST_ADVENTURE.read_text()

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

required_today_markers = [
    "accessibilityLabel={`${tool.label}. ${tool.caption}`}",
    'accessibilityLabel="Close outcome check-in"',
    "toolCard: {\n    flexGrow: 1,\n    flexBasis: 220,",
    "iconButton: {\n    minWidth: 44,\n    minHeight: 44,",
]
for marker in required_today_markers:
    if marker not in today:
        raise SystemExit(f"Today accessibility/reflow marker missing: {marker}")

if "width: '48%'" in today:
    raise SystemExit("Today relationship tools must not return to a fixed 48% card width")

required_first_adventure_markers = [
    "altRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 }",
    "altButton: {\n    flexGrow: 1,\n    flexBasis: 220,",
]
for marker in required_first_adventure_markers:
    if marker not in first_adventure:
        raise SystemExit(f"First Adventure accessibility/reflow marker missing: {marker}")

if "altRow: { flexDirection: 'row', gap: 10 }" in first_adventure:
    raise SystemExit("First Adventure alternate actions must remain wrap-capable")

for path in (ROOT / "apps/mobile/src").rglob("*.tsx"):
    text = path.read_text()
    for forbidden in ["allowFontScaling={false}", "maxFontSizeMultiplier={1}"]:
        if forbidden in text:
            raise SystemExit(
                f"native source disables user text scaling in {path.relative_to(ROOT)}: {forbidden}"
            )

print(
    "Native accessibility authority is explicit: navigation is content-driven, stack motion follows the "
    "OS reduced-motion preference, keyboard-sensitive capture surfaces are protected, Today and First "
    "Adventure avoid fixed narrow action/card layouts, icon-only outcome close is named with a 44-point "
    "target, and text scaling remains user-controlled. Physical-device VoiceOver, largest Dynamic Type, "
    "and TestFlight usability remain separate evidence gates."
)
