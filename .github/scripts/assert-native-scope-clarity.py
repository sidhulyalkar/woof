#!/usr/bin/env python3
"""Fail closed if native scope clarity drifts across Companion, Story, or Expedition."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMPANION = ROOT / "apps/mobile/src/screens/CompanionHomeScreen.tsx"
STORY = ROOT / "apps/mobile/src/screens/StoryScreen.tsx"
STORY_API = ROOT / "apps/mobile/src/api/story.ts"
HOUSEHOLDS_API = ROOT / "apps/mobile/src/api/households.ts"
EXPEDITION = ROOT / "apps/mobile/src/screens/ExpeditionScreen.tsx"
WORLD = ROOT / "apps/mobile/src/components/community/ExpeditionWorldView.tsx"
JOURNAL = ROOT / "apps/mobile/src/components/community/ExpeditionFieldJournalView.tsx"
DOC = ROOT / "docs/NATIVE_SCOPE_CLARITY_V1.md"


def fail(message: str) -> None:
    raise SystemExit(message)


def require(text: str, marker: str, label: str) -> None:
    if marker not in text:
        fail(f"{label} missing required marker: {marker}")


def reject(text: str, marker: str, label: str) -> None:
    if marker in text:
        fail(f"{label} contains forbidden marker: {marker}")


def require_count(text: str, marker: str, expected: int, label: str) -> None:
    actual = text.count(marker)
    if actual != expected:
        fail(f"{label} expected {expected} occurrences of {marker!r}, found {actual}")


def main() -> None:
    for path in [COMPANION, STORY, STORY_API, HOUSEHOLDS_API, EXPEDITION, WORLD, JOURNAL, DOC]:
        if not path.is_file():
            fail(f"missing scope clarity file: {path.relative_to(ROOT)}")

    companion = COMPANION.read_text()
    story = STORY.read_text()
    story_api = STORY_API.read_text()
    households_api = HOUSEHOLDS_API.read_text()
    expedition = EXPEDITION.read_text()
    world = WORLD.read_text()
    journal = JOURNAL.read_text()
    doc = DOC.read_text()

    # Companion is a first-class human-side starting point, never a synthetic pet scope.
    for marker in [
        "You can belong here before you have a dog.",
        "Dog-specific spaces stay private",
        "route: 'Expedition'",
        "route: 'Skillcraft'",
        "route: 'Packs'",
        "route: 'CommunityStandalone'",
        "pet relationship or unlock pet-specific Today, Compass, or Story.",
    ]:
        require(companion, marker, "Companion Home")
    for forbidden in [
        "adventureApi",
        "storyApi",
        "petsApi",
        "householdsApi",
        "useRelationshipScope",
        "selectedPetId",
    ]:
        reject(companion, forbidden, "Companion Home")

    # Story owns its own truthful scope: ALL is canonical default, pet filters are authorized
    # through household reads, and Today/Compass relationship preference is not inherited.
    require(story_api, "petId?: string", "Story API")
    require(
        households_api,
        "getMine: () => apiClient.get<HouseholdSnapshot[]>('/households/me')",
        "household API",
    )
    for marker in [
        "type StoryScope = 'ALL' | string",
        "useState<StoryScope>('ALL')",
        "householdsApi.getMine()",
        "scope === 'ALL' ? {} : { petId: scope }",
        "storyRequestRef",
        "requestId !== storyRequestRef.current",
        "scope !== selectedScopeRef.current",
        "dashboardScope === selectedScope",
        "All dogs",
        "All dogs is one authorized household view.",
        "accessibilityState={{ selected }}",
        "minHeight: 44",
    ]:
        require(story, marker, "Story scope")
    for forbidden in [
        "useRelationshipScope",
        "relationship-scope",
        "storyApi.post",
        "storyApi.put",
        "storyApi.patch",
        "storyApi.delete",
    ]:
        reject(story, forbidden, "Story scope")

    # Optional filter discovery may fail without blocking canonical Story, but stale pet-specific
    # filter state or stale same-scope Story data must not remain visible after authority reads fail.
    require(story, "setFilterPets([]);", "Story filter degradation")
    require(
        story,
        "Dog filters are unavailable. Story is using the server-authorized all-dogs view.",
        "Story filter degradation",
    )
    require_count(story, "setDashboard(null);", 2, "Story fail-closed data presentation")
    require_count(story, "setDashboardScope(null);", 2, "Story fail-closed data presentation")
    require(story, "setFilterError", "Story filter degradation")
    require(story, "setError('Story is unavailable right now.", "Story data degradation")

    # Field Journal is independently degradable from the live Expedition world. Refresh and Pack
    # requests are generation-bound so an older network result cannot erase a newer truth.
    for marker in [
        "const [worldError, setWorldError]",
        "const [journalError, setJournalError]",
        "loadRequestRef",
        "loadRequestId !== loadRequestRef.current",
        "packRequestRef",
        "selectedScopeRef.current !== joinedPack.id",
        "packLoadResult?.status === 'error'",
        "setWorldError(packLoadResult.message)",
        "journalRef.current",
        "journalError={journalError}",
        "Your shared world is still available",
    ]:
        require(expedition, marker, "Expedition failure isolation")
    reject(expedition, "unavailableWorld.push('field journal')", "Expedition failure isolation")
    reject(expedition, "unavailable.push('field journal')", "Expedition failure isolation")
    require(journal, "error?: string | null", "Field Journal failure presentation")
    require(expedition, "Showing your last verified pages.", "Field Journal stale-history authority")
    require(journal, "leave history blank rather than guess", "Field Journal fail-closed presentation")

    # The cooperative world keeps server calibration fields in the API but does not advertise
    # repeatable volume. Personal presence is binary and communal count is descriptive only.
    reject(world, "objective.total", "Expedition calm presentation")
    require_count(world, "objective.myContribution", 1, "Expedition calm presentation")
    require(world, "objective.myContribution > 0", "Expedition calm presentation")
    require_count(world, "objective.contributors", 1, "Expedition calm presentation")
    require(world, "There is no finish line to chase.", "Expedition calm presentation")
    require(world, "Presence matters more than repetition.", "Expedition calm presentation")
    require(world, "minHeight: 44", "Expedition scope accessibility")

    # The documentation must make all three scope classes explicit.
    for marker in [
        "Today + Compass",
        "Story",
        "Skillcraft + Expedition + Field Journal + Community",
        "human-side",
        "Local selection never grants authority",
        "Personal Expedition repetition is binary presence",
        "Journal failure does not poison the live world",
        "44pt",
    ]:
        require(doc, marker, "scope clarity documentation")

    print("Native scope clarity contract OK")


if __name__ == "__main__":
    main()
