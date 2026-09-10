#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(source: str, needle: str, message: str) -> None:
    if needle not in source:
        raise SystemExit(message)


def forbid(source: str, pattern: str, message: str) -> None:
    if re.search(pattern, source, flags=re.MULTILINE | re.DOTALL):
        raise SystemExit(message)


care = read("apps/api/src/care-events/care-events.service.ts")
compass = read("apps/mobile/src/screens/CompassScreen.tsx")
today = read("apps/mobile/src/screens/TodayScreen.tsx")
scope = read("apps/mobile/src/relationship/relationship-scope.ts")
selector = read("apps/mobile/src/components/relationship/RelationshipScopeBar.tsx")

# Cross-layer percentage contract. CareSummary owns a 0–100 opportunity percentage;
# native must not reinterpret that value as a 0–1 fraction.
require(
    care,
    "coverage: Math.min(100, recentDays * 25)",
    "CareSummary must retain the canonical 0–100 pathway coverage scale",
)
require(
    compass,
    "Math.min(100, item.coverage)",
    "Compass must clamp canonical coverage directly on the 0–100 scale",
)
forbid(
    compass,
    r"Math\.min\(\s*1\s*,\s*item\.coverage\s*\)",
    "Compass must not collapse 0–100 pathway coverage onto a 0–1 scale",
)
require(
    compass,
    "width: `${coveragePercent}%`",
    "Compass progress width must use the canonical percentage without multiplying again",
)

# Relationship selection is presentation state only. The usable set must come from
# authenticated household authority, and persisted IDs must be intersected with it.
require(scope, "householdsApi.getMine()", "Relationship scope must load household authority")
require(
    scope,
    "woof:selected-relationship-pet:v1:",
    "Relationship selection must be account-namespaced",
)
require(
    scope,
    "pets.some((pet) => pet.id === candidate)",
    "Stored pet IDs must be intersected with fresh authorized pets",
)
require(
    scope,
    "snapshot.pets.some((pet) => pet.id === petId)",
    "Interactive pet selection must reject IDs outside the authorized snapshot",
)
require(
    scope,
    "SecureStore.setItemAsync",
    "Relationship preference should survive a normal app restart",
)
require(
    scope,
    "Server household authority remains canonical",
    "Persistence failure must not become authorization authority",
)
forbid(
    scope,
    r"petsApi\.",
    "Relationship scope must not fall back to the legacy owner-only pets client",
)

# Slow household reads and slow dog-specific dashboard reads must not overwrite a
# newer account/pet choice.
require(scope, "let loadGeneration = 0", "Relationship authority refresh needs a generation guard")
require(
    scope,
    "generation !== loadGeneration",
    "Stale household refreshes must be rejected before updating shared state",
)
require(
    scope,
    "loadGeneration += 1",
    "An explicit pet choice must supersede an older household refresh",
)
require(today, "requestGenerationRef", "Today needs a stale-response generation guard")
require(compass, "requestGenerationRef", "Compass needs a stale-response generation guard")
require(
    today,
    "requestGeneration !== requestGenerationRef.current",
    "Today must reject stale cross-pet dashboard responses",
)
require(
    compass,
    "requestGeneration !== requestGenerationRef.current",
    "Compass must reject stale cross-pet dashboard responses",
)

# Pet-specific Adventure reads must always name the selected relationship.
require(
    today,
    "adventureApi.getMine(selectedPetId)",
    "Today must request the explicitly selected pet",
)
require(
    compass,
    "adventureApi.getMine(selectedPetId)",
    "Compass must request the explicitly selected pet",
)
require(
    today,
    "dashboard.pet.id === selectedPetId",
    "Today must suppress stale cross-pet dashboards",
)
require(
    compass,
    "dashboard.pet.id === selectedPetId",
    "Compass must suppress stale cross-pet dashboards",
)
require(today, "<RelationshipScopeBar", "Today must expose relationship scope")
require(compass, "<RelationshipScopeBar", "Compass must expose relationship scope")

# Selection controls need a comfortable mobile target and explicit accessibility state.
require(
    selector,
    "minHeight: 44",
    "Relationship selector controls must provide a 44pt minimum target",
)
require(
    selector,
    "accessibilityState={{ selected }}",
    "Relationship selector must expose selected state",
)
require(
    selector,
    "Each dog keeps a separate history.",
    "Multi-dog copy must make relationship separation explicit",
)

print("Native relationship context authority contract OK")
