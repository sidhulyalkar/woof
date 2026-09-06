#!/usr/bin/env python3

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
TRAIL_PATH = ROOT / "apps/mobile/src/game/adventure-trail.ts"
COMPASS_PATH = ROOT / "apps/mobile/src/screens/CompassScreen.tsx"
DOC_PATH = ROOT / "docs/NATIVE_ADVENTURE_TRAIL_V1.md"

trail = TRAIL_PATH.read_text()
compass = COMPASS_PATH.read_text()
doc = DOC_PATH.read_text()


def require(source: str, marker: str, message: str) -> None:
    if marker not in source:
        raise SystemExit(message)


require(
    trail,
    "adventure-trail-presentation-v1",
    "Adventure Trail presentation policy version is missing",
)

pathway_match = re.search(
    r"export const TRAIL_PATHWAYS = \[(.*?)\] as const satisfies readonly WellbeingPathway\[\];",
    trail,
    re.DOTALL,
)
if pathway_match is None:
    raise SystemExit("Unable to parse Adventure Trail pathway authority")

pathways = re.findall(r"'([A-Z_]+)'", pathway_match.group(1))
expected_pathways = [
    "MOVE",
    "EXPLORE",
    "ENRICH",
    "LEARN",
    "CONNECT",
    "RECOVER",
    "BOND",
]
if pathways != expected_pathways:
    raise SystemExit(f"Adventure Trail pathways drifted: {pathways!r}")
if "CARE" in pathways:
    raise SystemExit("CARE must remain outside Adventure Trail discovery collection")

thresholds = [int(value) for value in re.findall(r"minBondXp: (\d+)", trail)]
if len(thresholds) != 6:
    raise SystemExit(f"Expected six Adventure Trail chapters, found {len(thresholds)}")
if thresholds[0] != 0:
    raise SystemExit("Adventure Trail must begin at zero Bond XP")
if thresholds != sorted(set(thresholds)):
    raise SystemExit("Adventure Trail chapter thresholds must be unique and strictly increasing")

for marker in ("dashboard.bondXp", "dashboard.compass", "dashboard.rhythm"):
    require(trail, marker, f"Adventure Trail must derive from canonical dashboard field: {marker}")

for forbidden in (
    "apiClient",
    "adventureApi",
    ".post(",
    ".put(",
    ".delete(",
    "Math.random(",
    "Date.now(",
    "new Date(",
):
    if forbidden in trail:
        raise SystemExit(f"Adventure Trail presentation policy must stay deterministic/read-only: {forbidden}")

for marker in (
    "deriveAdventureTrail(dashboard)",
    "ADVENTURE TRAIL",
    "Discovery stamps",
    "CARE stays visible in the Compass below, but it is intentionally outside this collection",
    "Missing a day never resets Rhythm",
    "They never unlock care or change",
):
    require(compass, marker, f"Native Adventure Trail UI boundary missing: {marker}")

for marker in (
    "The human gets a game-shaped sense of unfolding progress.",
    "The dog keeps the right to have an ordinary day.",
    "`CARE` is intentionally excluded from the collection layer.",
    "It does not create a daily streak",
):
    require(doc, marker, f"Native Adventure Trail documentation boundary missing: {marker}")

print("Native Adventure Trail source contract passed")
