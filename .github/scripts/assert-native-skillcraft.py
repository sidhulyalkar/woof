#!/usr/bin/env python3

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MOBILE_API = ROOT / "apps/mobile/src/api/social-adventure.ts"
SCREEN = ROOT / "apps/mobile/src/screens/SkillcraftScreen.tsx"
NAV = ROOT / "apps/mobile/src/navigation/AppNavigator.tsx"
COMPANION = ROOT / "apps/mobile/src/screens/CompanionHomeScreen.tsx"
FEED = ROOT / "apps/mobile/src/screens/FeedScreen.tsx"
SERVER_POLICY = ROOT / "apps/api/src/social-adventure/social-adventure.policy.ts"
SERVER_ARCADE = ROOT / "apps/api/src/social-adventure/social-adventure.arcade.ts"
SERVER_SERVICE = ROOT / "apps/api/src/social-adventure/social-adventure.service.ts"
DOC = ROOT / "docs/NATIVE_SKILLCRAFT_V1.md"

mobile_api = MOBILE_API.read_text()
screen = SCREEN.read_text()
nav = NAV.read_text()
companion = COMPANION.read_text()
feed = FEED.read_text()
server_policy = SERVER_POLICY.read_text()
server_arcade = SERVER_ARCADE.read_text()
server_service = SERVER_SERVICE.read_text()
doc = DOC.read_text()


def require(source: str, marker: str, message: str) -> None:
    if marker not in source:
        raise SystemExit(message)


challenge_match = re.search(
    r"export const HUMAN_SKILL_CHALLENGES = \[(.*?)\] as const;",
    server_policy,
    re.DOTALL,
)
if challenge_match is None:
    raise SystemExit("Unable to parse server Human Skill challenge authority")

challenges = re.findall(r"'([A-Z_]+)'", challenge_match.group(1))
expected_challenges = [
    "MAKE_IT_EASIER",
    "CATCH_THE_GOOD",
    "PAIRING_LAB",
    "MARKER_TIMING",
]
if challenges != expected_challenges:
    raise SystemExit(f"Human Skill challenge set drifted: {challenges!r}")

for marker in (
    "'/social-adventure/arcade'",
    "`/social-adventure/arcade/${challengeKey}/attempts`",
    "`/social-adventure/arcade/attempts/${attemptId}/complete`",
    "'/social-adventure/shares'",
    "sourceType: 'HUMAN_SKILL_ATTEMPT'",
    "visibility: 'PUBLIC'",
):
    require(mobile_api, marker, f"Native Skillcraft API contract missing: {marker}")

for marker in (
    "socialAdventureApi.arcade()",
    "socialAdventureApi.startArcadeAttempt",
    "socialAdventureApi.completeArcadeAttempt",
    "socialAdventureApi.shareSkillAttempt",
    "Breadth counts once. Grinding does not.",
    "Practice scores stay personal feedback.",
    "retries and higher scores add no rank.",
    "The dog does not have to perform for you to play.",
    "Sharing is optional and publishes this human practice moment only.",
    "reactions do not increase your rank.",
    "A game is not training authority.",
    "onPress={() => void shareResult()}",
):
    require(screen, marker, f"Native Skillcraft UI boundary missing: {marker}")

if screen.count("socialAdventureApi.shareSkillAttempt") != 1:
    raise SystemExit("Skillcraft sharing must remain one explicit post-completion action")

for forbidden in (
    "correctOptionId",
    "scoreArcadeResponse",
    "quieter_context",
    "mark_settle",
    "sound_then_good",
    "../api/adventure",
    "../api/pets",
    "../api/activities",
    "../api/daily-signals",
    "careEvent",
    "petId",
):
    if forbidden in screen or forbidden in mobile_api:
        raise SystemExit(f"Native Skillcraft crossed an authority boundary: {forbidden}")

for marker in (
    "correctOptionId",
    "scoreArcadeResponse",
    "quieter_context",
    "mark_settle",
    "sound_then_good",
):
    require(server_arcade, marker, f"Server Arcade scoring authority missing: {marker}")

for marker in (
    "async getArcade(userId: string)",
    "async startHumanSkillAttempt(userId: string, challengeKey: string)",
    "async completeHumanSkillAttempt(",
    "private async getBestHumanSkillScores(userId: string)",
    "const season = currentUtcSeason();",
    "completed_at >= ${season.startsAt}",
    "completed_at < ${season.endsAt}",
):
    require(server_service, marker, f"Server Skillcraft authority missing: {marker}")

if nav.count('name="Skillcraft"') != 2:
    raise SystemExit("Skillcraft must remain registered in both Guardian and Companion navigators")
require(nav, "Skillcraft: undefined;", "Skillcraft route type is missing")
require(companion, "route: 'Skillcraft'", "Companion mode must expose pet-independent Skillcraft")
require(feed, "navigation.navigate('Skillcraft')", "Community must link into Skillcraft")

for marker in (
    "A dog does not need to perform for the human to practice.",
    "weekly breadth, not grind",
    "replaying the same room does not add another breadth unit",
    "automatically share a result",
    "Pet relationships still control pet authority.",
    "the human gets the game; the dog keeps the right to have an ordinary day.",
):
    require(doc.lower(), marker.lower(), f"Native Skillcraft documentation boundary missing: {marker}")

print("Native Skillcraft source contract passed")
