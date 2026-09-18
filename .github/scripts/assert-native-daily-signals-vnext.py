#!/usr/bin/env python3
"""Fail closed when native Daily Signals vNext sparse-capture semantics drift."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAV = ROOT / "apps/mobile/src/navigation/AppNavigator.tsx"
TODAY = ROOT / "apps/mobile/src/screens/TodayScreen.tsx"
DAILY = ROOT / "apps/mobile/src/screens/DailySignalsScreen.tsx"

for path in [NAV, TODAY, DAILY]:
    if not path.is_file():
        raise SystemExit(f"Daily Signals vNext source missing: {path.relative_to(ROOT)}")

nav = NAV.read_text()
today = TODAY.read_text()
daily = DAILY.read_text()

for marker in [
    "DailySignals: { preferredPetId?: string } | undefined;",
    "preferredPetId={route.params?.preferredPetId}",
    "onDone={() => navigation.goBack()}",
]:
    if marker not in nav:
        raise SystemExit(f"Daily Signals navigation-context contract drifted: {marker}")

for marker in [
    "navigation.navigate('DailySignals'",
    "preferredPetId: selectedPetId",
]:
    if marker not in today:
        raise SystemExit(f"Today -> Daily Signals relationship handoff drifted: {marker}")

required_daily = [
    "Anything different today?",
    "Skipping the rest leaves it unknown.",
    "resolveSelectedContextKey",
    "const preferred = contexts.filter((context) => context.petId === preferredPetId);",
    "if (preferred.length === 1) return contextKey(preferred[0]!);",
    "if (contexts.length === 1) return contextKey(contexts[0]!);",
    "Not reported",
    "expandedDimension",
    "Leave unreported",
    "Add a private note",
    "Nothing to add today",
    "answeredCount === 0",
    "signals: answers",
    "Unreported stays unknown.",
    "“Not sure” stays uncertainty",
    "minHeight: 44",
    "flexWrap: 'wrap'",
]
missing = [marker for marker in required_daily if marker not in daily]
if missing:
    raise SystemExit(f"Daily Signals sparse-capture contract drifted: {missing}")

for forbidden in [
    "selectedIndex",
    "contexts[0]",
    "4 of 6",
    "5 of 6",
    "6 of 6",
    "completion percentage",
    "streak",
    "dimensionCard:",
]:
    if forbidden in daily:
        raise SystemExit(f"Daily Signals vNext regressed to checklist/order semantics: {forbidden}")

if "const [answers, setAnswers] = useState<DailySignalsAnswers>({});" not in daily:
    raise SystemExit("Daily Signals answers must start sparse/empty; untouched cannot imply USUAL")

if "disabled={saving || answeredCount === 0 || !selected.timezone}" not in daily:
    raise SystemExit("note-only/empty Daily Signals must not become canonical evidence")

print(
    "Native Daily Signals vNext sparse-capture authority is explicit: untouched dimensions stay unknown, "
    "only a uniquely authorized relationship context is preselected, note-only capture cannot save, "
    "large-text choices wrap with 44-point targets, and leaving without observations writes nothing. "
    "Correction/supersession authority remains a separate release gate."
)
