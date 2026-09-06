#!/usr/bin/env python3
"""Static release contract for native First Adventure v1.

This verifies authority boundaries and Web/native ontology parity. Runtime,
typing, lint, and native-build qualification belong to the workflows that
invoke or accompany this contract.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def string_values(text: str, constant: str) -> set[str]:
    match = re.search(
        rf"export const {re.escape(constant)}\s*=\s*\[(.*?)\]\s*as const;",
        text,
        flags=re.S,
    )
    if not match:
        raise AssertionError(f"could not locate {constant}")
    return set(re.findall(r"['\"]([A-Z0-9_]+)['\"]", match.group(1)))


nav = read("apps/mobile/src/navigation/AppNavigator.tsx")
companion = read("apps/mobile/src/api/companion.ts")
auth = read("apps/mobile/src/api/auth.ts")
pets = read("apps/mobile/src/api/pets.ts")
profile = read("apps/mobile/src/api/adaptive-profile.ts")
recovery = read("apps/mobile/src/onboarding/recovery.ts")
register_dto = read("apps/api/src/auth/dto/register.dto.ts")
native_questions = read("apps/mobile/src/onboarding/first-adventure.ts")
web_questions = read("apps/web/src/lib/onboarding/first-adventure.ts")
first_adventure = read("apps/mobile/src/screens/FirstAdventureScreen.tsx")
companion_home = read("apps/mobile/src/screens/CompanionHomeScreen.tsx")

# Server-owned landing authority. Authentication alone must never unlock pet UI.
for landing in ("NEEDS_MODE", "NEEDS_PET_SETUP", "PET_TODAY", "COMPANION_TODAY"):
    require(companion, f"'{landing}'", f"Companion landing {landing}")
    require(nav, f"state.landing === '{landing}'", f"native router branch {landing}")
require(nav, "next.landing === 'PET_TODAY'", "server-confirmed guardian recovery cleanup")
require(nav, "companionApi.state()", "server Companion-state resolution")
require(nav, "unsupported account mode", "unknown landing fail-closed path")
require(nav, "Pet-specific surfaces stay closed", "fail-closed pet-surface copy")

# Registration and first-pet creation reuse the existing server replay contracts.
require(register_dto, "@IsUUID()", "server registration UUID validation")
require(register_dto, "registrationKey?: string", "server registration replay field")
require(recovery, "UUID_V4", "native registration UUID validation")
require(recovery, "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx", "native UUID-v4 generator")
require(recovery, "registrationReplayKey()", "native registration UUID generation")
require(auth, "registrationKey: recovery.registrationKey", "registration replay key transport")
require(auth, "await clearRegistrationRecovery()", "registration recovery cleanup")
require(pets, "creationKey: string", "pet replay identity type")
require(pets, "createDog", "minimal replay-safe dog create")
require(recovery, "ambiguous?: boolean", "ambiguous write persistence")
require(recovery, "markPetCreationAmbiguous", "ambiguous write marker")
require(first_adventure, "await markPetCreationAmbiguous(ambiguous)", "ambiguous create persistence use")
require(first_adventure, "if (ambiguousCreate)", "ambiguous-create mode guard")
require(first_adventure, "Retry exact create", "exact-retry user path")
require(first_adventure, "Check server state first", "authority recheck user path")
require(first_adventure, "editable={!creating && !ambiguousCreate}", "frozen ambiguous identity fields")
require(first_adventure, "modeSwitchDisabled", "ambiguous mode-switch lock")

# Optional profile evidence cannot become an access gate or a reward surface.
require(first_adventure, "Promise.allSettled", "non-blocking optional evidence writes")
require(first_adventure, "Skip personalization and open Today", "skip personalization path")
require(
    first_adventure,
    "Skipping never reduces access, rewards, or relationship status",
    "no-skip-penalty law",
)
require(profile, "/questions/respond", "canonical Adaptive Profile response endpoint")
require(native_questions, "outcome: 'NOT_SURE'", "explicit uncertainty semantics")
require(native_questions, "outcome: 'SKIPPED'", "explicit skip semantics")

# Native must share the exact question IDs and closed answer vocabulary with Web.
for constant in (
    "FIRST_ADVENTURE_GOALS",
    "FIRST_ADVENTURE_TIME_BUDGETS",
    "FIRST_ADVENTURE_EFFORT_LEVELS",
    "FIRST_ADVENTURE_SOCIAL_COMFORT",
):
    native = string_values(native_questions, constant)
    web = string_values(web_questions, constant)
    if native != web:
        raise AssertionError(f"{constant} drift: native={sorted(native)} web={sorted(web)}")

question_ids = (
    "profile-owner-goals-v1",
    "profile-owner-time-budget-v1",
    "profile-owner-effort-v1",
    "profile-dog-social-comfort-v1",
)
for question_id in question_ids:
    require(native_questions, question_id, f"native question id {question_id}")
    require(web_questions, question_id, f"web question id {question_id}")

# Petless Companion mode is a truthful first-class route, not a broken pet Today.
require(companion_home, "You do not need to invent a dog", "petless Companion framing")
require(companion_home, "CommunityStandalone", "petless Community route")
require(companion_home, "Presentation is not authority", "mode/authority separation")

print("native First Adventure authority contract: PASS")
