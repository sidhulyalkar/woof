#!/usr/bin/env python3
"""Fail closed if native Expedition presentation invents client-side game authority."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MOBILE_API = ROOT / "apps/mobile/src/api/expeditions.ts"
SCREEN = ROOT / "apps/mobile/src/screens/ExpeditionScreen.tsx"
WORLD = ROOT / "apps/mobile/src/components/community/ExpeditionWorldView.tsx"
FEED = ROOT / "apps/mobile/src/screens/FeedScreen.tsx"
COMMUNITY = ROOT / "apps/mobile/src/components/community/SocialAdventureCommunityView.tsx"
NAV = ROOT / "apps/mobile/src/navigation/AppNavigator.tsx"
SERVER_CONTROLLER = ROOT / "apps/api/src/expeditions/expeditions.controller.ts"
SERVER_POLICY = ROOT / "apps/api/src/expeditions/expeditions.policy.ts"
DOC = ROOT / "docs/NATIVE_EXPEDITION_WORLD_V1.md"


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
    required = [
        MOBILE_API,
        SCREEN,
        WORLD,
        FEED,
        COMMUNITY,
        NAV,
        SERVER_CONTROLLER,
        SERVER_POLICY,
        DOC,
    ]
    for path in required:
        if not path.is_file():
            fail(f"missing required native Expedition file: {path.relative_to(ROOT)}")

    mobile_api = MOBILE_API.read_text()
    screen = SCREEN.read_text()
    world = WORLD.read_text()
    feed = FEED.read_text()
    community = COMMUNITY.read_text()
    nav = NAV.read_text()
    server_controller = SERVER_CONTROLLER.read_text()
    server_policy = SERVER_POLICY.read_text()
    doc = DOC.read_text()

    # Native Expedition has exactly two read paths and no contribution mutation surface.
    for marker in [
        "'/expeditions/global'",
        "`/expeditions/packs/${packId}`",
        "target: null",
        "status: 'CALIBRATING'",
        "'SNIFF_EXPLORE' | 'RECOVERY_COUNTS' | 'READ_THE_ROOM'",
    ]:
        require(mobile_api, marker, "native Expedition API")

    require_count(mobile_api, "apiClient.get<ExpeditionProjection>", 2, "native Expedition API")
    for forbidden in [
        "apiClient.post",
        "apiClient.put",
        "apiClient.patch",
        "apiClient.delete",
        "contribute:",
        "complete:",
        "increment:",
    ]:
        reject(mobile_api, forbidden, "native Expedition API")

    # Pack reads must be downstream of a server-confirmed joined Pack and identity-bound response.
    for marker in [
        "expeditionApi.global()",
        "socialAdventureApi.packs()",
        "pack.id === packId && pack.joined",
        "expeditionApi.pack(joinedPack.id)",
        "response.scope !== 'PACK' || response.pack?.id !== joinedPack.id",
        "packRequestRef",
        "requestId !== packRequestRef.current",
        "selectedScopeRef",
        "Promise.allSettled([",
        "Woof will only open Expedition views for Packs the server says you joined.",
        "Woof did not infer missing progress or membership.",
    ]:
        require(screen, marker, "native Expedition screen")

    require_count(screen, "expeditionApi.pack(", 1, "native Expedition screen")
    for forbidden in [
        "globalLeaderboard()",
        "packLeaderboard(",
        "adventureApi.",
        "storyApi.",
        "expo-location",
        "getCurrentPosition",
        "watchPosition",
    ]:
        reject(screen, forbidden, "native Expedition screen")

    # The renderer may display each numeric authority field, but may not repeatedly consume it
    # to create shadow geometry, unlocks, or local scoring. myContribution gets one additional
    # binary use for the explicitly documented Your-mark treatment.
    require_count(world, "objective.total", 1, "native Expedition world")
    require_count(world, "objective.contributors", 1, "native Expedition world")
    require_count(world, "objective.myContribution", 2, "native Expedition world")
    require(world, "objective.myContribution > 0", "native Expedition world")

    for marker in [
        "Wandering Grove",
        "Resting Hollow",
        "Signal Observatory",
        "objective.status === 'CALIBRATING'",
        "Calibrating, no completion target yet.",
        "No completion bar. This shared scene is not a checklist.",
        "Landmarks always exist. Counts show server-issued evidence, not unlock levels.",
        "The world is the game. Your dog is not.",
        "CARE, health state, distance, duration, intensity, missed days, likes, rankings",
    ]:
        require(world, marker, "native Expedition world")

    for forbidden in [
        "Math.min(",
        "Math.max(",
        "* 100",
        "ProgressBar",
        "<Progress",
        "progress:",
        "percentage",
        "percentComplete",
        "objective.total /",
        "objective.contributors /",
        "objective.myContribution /",
        "objective.target ??",
        "objective.target ||",
        "objective.target ?",
        "unlockThreshold",
        "levelThreshold",
    ]:
        reject(world + screen, forbidden, "native Expedition projection surface")

    # Community is a portal to cooperative play, not a hidden embedding in league arithmetic.
    require(feed, "onOpenExpedition={() => navigation.navigate('Expedition')}", "native Community screen")
    require(community, "onOpenExpedition: () => void", "native Community presentation")
    require(community, 'label="Expedition" onPress={props.onOpenExpedition}', "native Community presentation")

    # Both guardian and Companion stacks expose the same human-side Expedition surface.
    require(nav, "Expedition: undefined", "native navigation")
    require(nav, "import ExpeditionScreen from '../screens/ExpeditionScreen'", "native navigation")
    require_count(nav, 'name="Expedition"', 2, "native navigation")
    require_count(nav, "component={ExpeditionScreen}", 2, "native navigation")

    # Server policy remains the source of what counts and continues to expose calibration.
    for marker in [
        "@Get('global')",
        "@Get('packs/:packId')",
    ]:
        require(server_controller, marker, "server Expedition controller")

    for marker in [
        "EXPEDITION_POLICY_VERSION = 'expedition-authority-v1'",
        "EXPEDITION_ELIGIBLE_PATHWAYS = ['EXPLORE', 'ENRICH', 'RECOVER']",
        "perContributorCap",
        "perCategoryCap",
        "score magnitude does not",
    ]:
        require(server_policy, marker, "server Expedition policy")

    for marker in [
        "There is no native Expedition contribution mutation.",
        "Calibration means no meter",
        "Every landmark exists from the beginning.",
        "The scene geometry is intentionally static with respect to totals.",
        "server returned `joined: true`",
        "Human Skill practice-score magnitude does not become Expedition progress",
        "Companion-mode users can open the same Expedition surface",
        "does not persist completed Expedition seasons into Story",
    ]:
        require(doc, marker, "native Expedition documentation")

    for forbidden in [
        "client-authoritative target",
        "daily streak reward",
        "pet performance rank",
    ]:
        reject(doc.lower(), forbidden.lower(), "native Expedition documentation")

    print("Native Expedition World authority contract OK")


if __name__ == "__main__":
    main()
