#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


nav = read("apps/mobile/src/navigation/AppNavigator.tsx")
today = read("apps/mobile/src/screens/TodayScreen.tsx")
community = read("apps/mobile/src/screens/FeedScreen.tsx")
daily = read("apps/mobile/src/screens/DailySignalsScreen.tsx")
adventure = read("apps/mobile/src/api/adventure.ts")
story = read("apps/mobile/src/api/story.ts")
households = read("apps/mobile/src/api/households.ts")
intelligence = read("apps/mobile/src/api/intelligence.ts")

for marker in [
    'Today: undefined',
    'Compass: undefined',
    'Story: undefined',
    'Community: undefined',
    '<Tab.Screen name="Today"',
    '<Tab.Screen name="Compass"',
    '<Tab.Screen name="Story"',
    '<Tab.Screen name="Community"',
]:
    if marker not in nav:
        raise SystemExit(f"mobile navigation spine marker missing: {marker}")

primary_tabs = re.findall(r'<Tab\.Screen\s+name="([^"]+)"', nav)
if primary_tabs != ["Today", "Compass", "Story", "Community"]:
    raise SystemExit(f"mobile primary navigation drifted: {primary_tabs}")

for legacy_primary in ['name="Feed"', 'name="Map"', 'name="Events"', 'name="Goals"', 'name="Profile"']:
    if f"<Tab.Screen {legacy_primary}" in nav:
        raise SystemExit(f"legacy capability returned as a primary tab: {legacy_primary}")

for marker in [
    "Woof recommends, you choose.",
    "Making it easier, changing your mind, or stopping when your dog is done",
    "Save what Woof should learn",
    "Relationship tools",
]:
    if marker not in today:
        raise SystemExit(f"relationship-first Today marker missing: {marker}")

for forbidden in ["PostDetail", "CreatePost"]:
    if forbidden in community:
        raise SystemExit(f"phantom mobile community route returned: {forbidden}")

for marker in [
    "This is not a diagnosis or health score.",
    "'UNSURE'",
    "already recorded for this dog and household day",
    "householdId: selected.householdId",
]:
    if marker not in daily:
        raise SystemExit(f"Daily Signals authority marker missing: {marker}")

for source, marker in [
    (adventure, "'/adventure/me'"),
    (story, "'/story'"),
    (households, "'/households/me'"),
    (intelligence, "'/intelligence/daily-signals'"),
]:
    if marker not in source:
        raise SystemExit(f"canonical mobile API marker missing: {marker}")

print("mobile dogOS convergence source contract: OK")
