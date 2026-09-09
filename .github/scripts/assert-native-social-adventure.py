#!/usr/bin/env python3
"""Fail closed if native Social Adventure drifts from server-owned game authority."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MOBILE_API = ROOT / "apps/mobile/src/api/social-adventure.ts"
FEED = ROOT / "apps/mobile/src/screens/FeedScreen.tsx"
COMMUNITY_VIEW = ROOT / "apps/mobile/src/components/community/SocialAdventureCommunityView.tsx"
PACKS = ROOT / "apps/mobile/src/screens/PacksScreen.tsx"
PACKS_VIEW = ROOT / "apps/mobile/src/components/community/SocialAdventurePacksView.tsx"
NAV = ROOT / "apps/mobile/src/navigation/AppNavigator.tsx"
SERVER_POLICY = ROOT / "apps/api/src/social-adventure/social-adventure.policy.ts"
SERVER_SERVICE = ROOT / "apps/api/src/social-adventure/social-adventure.service.ts"
SERVER_DTO = ROOT / "apps/api/src/social-adventure/dto/social-adventure.dto.ts"
DOC = ROOT / "docs/NATIVE_SOCIAL_ADVENTURE_V1.md"


def fail(message: str) -> None:
    raise SystemExit(message)


def require(text: str, marker: str, label: str) -> None:
    if marker not in text:
        fail(f"{label} missing required marker: {marker}")


def reject(text: str, marker: str, label: str) -> None:
    if marker in text:
        fail(f"{label} contains forbidden marker: {marker}")


def main() -> None:
    required = [
        MOBILE_API,
        FEED,
        COMMUNITY_VIEW,
        PACKS,
        PACKS_VIEW,
        NAV,
        SERVER_POLICY,
        SERVER_SERVICE,
        SERVER_DTO,
        DOC,
    ]
    for path in required:
        if not path.is_file():
            fail(f"missing required native Social Adventure file: {path.relative_to(ROOT)}")

    mobile_api = MOBILE_API.read_text()
    feed = FEED.read_text()
    community = COMMUNITY_VIEW.read_text()
    packs = PACKS.read_text()
    packs_view = PACKS_VIEW.read_text()
    packs_surface = packs + packs_view
    nav = NAV.read_text()
    server_policy = SERVER_POLICY.read_text()
    server_service = SERVER_SERVICE.read_text()
    server_dto = SERVER_DTO.read_text()
    doc = DOC.read_text()

    for marker in [
        "'/social-adventure/me'",
        "'/social-adventure/preferences'",
        "'/social-adventure/leaderboard/global'",
        "'/social-adventure/feed'",
        "'/social-adventure/packs'",
        "'/social-adventure/arcade'",
        "HUMAN_SKILL_ATTEMPT",
        "cohortReady: boolean",
        "localMinimumCohort",
    ]:
        require(mobile_api, marker, "mobile Social Adventure API")

    for marker in [
        "socialAdventureApi.feed()",
        "socialAdventureApi.getMine()",
        "socialAdventureApi.globalLeaderboard()",
        "socialAdventureApi.updatePreferences(next)",
        "socialAdventureApi.addReaction",
        "socialAdventureApi.removeReaction",
        "Promise.allSettled([",
        "Your visibility preference was saved by the server",
        "Previously loaded server content is still shown where available",
        "navigation.navigate('Skillcraft')",
        "navigation.navigate('Packs')",
    ]:
        require(feed, marker, "native Community screen")

    if feed.count("socialAdventureApi.updatePreferences(next)") != 1:
        fail("native Community must have exactly one explicit global-visibility mutation path")
    reject(feed, "Promise.all([", "native Community screen")
    reject(feed, "Your prior setting remains authoritative.", "native Community screen")

    for marker in [
        "You compete. Your dog does not.",
        "Your score is private by default.",
        "Reactions build culture, not rank.",
        "Nothing posts automatically.",
        "An empty podium is allowed.",
        "Make my rank private",
        "Join global league",
        "Nice read",
        "Good call",
        "Trying this",
        "Adventure inspiration",
        "Cheer",
    ]:
        require(community, marker, "native Community presentation")

    for forbidden in [
        "../api/social",
        "socialApi.",
        "totalLikes",
        "commentsCount",
        "petId:",
        "petId?:",
        ".sort(",
        "sort((",
    ]:
        reject(feed + community + mobile_api, forbidden, "native Community authority surface")

    for marker in [
        "socialAdventureApi.packs()",
        "socialAdventureApi.createPack",
        "socialAdventureApi.joinPack",
        "socialAdventureApi.leavePack",
        "socialAdventureApi.packLeaderboard",
        "leaderboardRequestRef",
        "requestId !== leaderboardRequestRef.current",
        "response.pack.id !== packId",
        "leaderboard?.pack.id === selectedPack?.id",
        "Choose a broad community label, not a coordinate or precise place.",
        "Woof will not estimate a rank locally.",
    ]:
        require(packs_surface, marker, "native Packs surface")

    for marker in [
        "leaderboard.cohortReady",
        "leaderboard.minimumCohort",
        "catalog.locationContract",
        "pack.role === 'OWNER'",
    ]:
        require(packs_surface, marker, "native Pack privacy boundary")

    for forbidden in [
        "expo-location",
        "Location.request",
        "getCurrentPosition",
        "watchPosition",
        "navigator.geolocation",
        ".sort(",
        "sort((",
    ]:
        reject(packs_surface, forbidden, "native Packs authority surface")

    require(nav, "Community: undefined", "native navigation")
    require(nav, "Packs: undefined", "native navigation")
    require(nav, "name=\"Community\"", "native navigation")
    require(nav, "name=\"Packs\"", "native navigation")

    for marker in [
        "SOCIAL_ADVENTURE_SCORE_POLICY_VERSION",
        "humanSkill",
        "adventureVariety",
        "GLOBAL_LEADERBOARD_OPT_IN",
    ]:
        require(server_policy + server_service, marker, "server Social Adventure authority")

    for forbidden in [
        "steps",
        "distance",
        "mileage",
        "duration",
        "likeCount",
        "commentCount",
        "streak",
        "healthScore",
    ]:
        reject(server_policy.lower(), forbidden.lower(), "server Social Adventure score policy")

    require(server_dto, "v1 enforces slug syntax and length only", "server Pack locality contract")
    require(server_dto, "clients must not collect or submit device coordinates", "server Pack locality contract")
    reject(
        server_dto,
        "Never an address, coordinate, or route trace.",
        "server Pack locality contract",
    )

    for marker in [
        "You compete. Your dog does not.",
        "Community reads degrade independently.",
        "server's mutation response as the immediate authority",
        "user-supplied broad-area `regionKey`",
        "does **not** semantically prove",
        "Selection changes invalidate older in-flight requests",
        "Pack leaderboard responses are request/Pack-bound",
        "Expeditions are a candidate product layer, not the automatic next release.",
        "production deployment, physical-device use, restore evidence, and a small owner pilot",
    ]:
        require(doc, marker, "native Social Adventure documentation")

    for forbidden in [
        "auto-opt-in",
        "client-derived rank",
        "device geolocation is required",
    ]:
        reject(doc.lower(), forbidden.lower(), "native Social Adventure documentation")

    print("Native Social Adventure authority contract OK")


if __name__ == "__main__":
    main()
