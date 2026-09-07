#!/usr/bin/env python3

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MOBILE_API = ROOT / "apps/mobile/src/api/social-adventure.ts"
FEED = ROOT / "apps/mobile/src/screens/FeedScreen.tsx"
PACKS = ROOT / "apps/mobile/src/screens/PacksScreen.tsx"
NAV = ROOT / "apps/mobile/src/navigation/AppNavigator.tsx"
SERVER_POLICY = ROOT / "apps/api/src/social-adventure/social-adventure.policy.ts"
SERVER_SERVICE = ROOT / "apps/api/src/social-adventure/social-adventure.service.ts"
DOC = ROOT / "docs/NATIVE_SOCIAL_ADVENTURE_V1.md"

mobile_api = MOBILE_API.read_text()
feed = FEED.read_text()
packs = PACKS.read_text()
nav = NAV.read_text()
server_policy = SERVER_POLICY.read_text()
server_service = SERVER_SERVICE.read_text()
doc = DOC.read_text()


def require(source: str, marker: str, message: str) -> None:
    if marker not in source:
        raise SystemExit(message)


def normalized(source: str) -> str:
    return re.sub(r"\s+", " ", source).strip()


for marker in (
    "'/social-adventure/me'",
    "'/social-adventure/preferences'",
    "'/social-adventure/leaderboard/global'",
    "'/social-adventure/feed'",
    "`/social-adventure/shares/${shareId}/reactions`",
    "`/social-adventure/shares/${shareId}/reactions/${reaction}`",
    "'/social-adventure/packs'",
    "`/social-adventure/packs/${packId}/join`",
    "`/social-adventure/packs/${packId}/membership`",
    "`/social-adventure/packs/${packId}/leaderboard`",
):
    require(mobile_api, marker, f"Native Social Adventure API contract missing: {marker}")

reaction_match = re.search(
    r"export type SocialAdventureReaction =\s*(.*?);",
    mobile_api,
    re.DOTALL,
)
if reaction_match is None:
    raise SystemExit("Unable to parse native Social Adventure reaction authority")

reactions = re.findall(r"'([A-Z_]+)'", reaction_match.group(1))
expected_reactions = [
    "NICE_READ",
    "GOOD_CALL",
    "TRYING_THIS",
    "ADVENTURE_INSPIRATION",
    "CHEER",
]
if reactions != expected_reactions:
    raise SystemExit(f"Native semantic reaction set drifted: {reactions!r}")

post_match = re.search(
    r"export type SocialAdventurePost = \{(.*?)\n\};",
    mobile_api,
    re.DOTALL,
)
if post_match is None:
    raise SystemExit("Unable to parse native Social Adventure post type")
for forbidden in ("petId", "likesCount", "commentsCount"):
    if forbidden in post_match.group(1):
        raise SystemExit(f"Native feed reintroduced unnecessary pet/popularity authority: {forbidden}")

if "from '../api/social'" in feed or "socialApi." in feed:
    raise SystemExit("Native Community must not use the legacy social feed authority")

for marker in (
    "socialAdventureApi.feed()",
    "socialAdventureApi.getMine()",
    "socialAdventureApi.globalLeaderboard()",
    "socialAdventureApi.updatePreferences(next)",
    "socialAdventureApi.addReaction",
    "socialAdventureApi.removeReaction",
    "const toggleGlobalVisibility = async () =>",
    "onPress={() => void toggleGlobalVisibility()}",
    "navigation.navigate('Packs')",
):
    require(feed, marker, f"Native Community authority missing: {marker}")

if feed.count("socialAdventureApi.updatePreferences(next)") != 1:
    raise SystemExit("Global leaderboard visibility must change through one explicit UI action")

feed_copy = normalized(feed)
for marker in (
    "You compete. Your dog doesn't.",
    "Your score is private by default.",
    "Reactions build culture, not rank.",
    "Nothing posts automatically.",
    "An empty podium is allowed.",
    "Make my rank private",
    "Join global league",
):
    require(feed_copy, marker, f"Native Community boundary copy missing: {marker}")

for marker in (
    "socialAdventureApi.packs()",
    "socialAdventureApi.createPack",
    "socialAdventureApi.joinPack",
    "socialAdventureApi.leavePack",
    "socialAdventureApi.packLeaderboard",
    "leaderboard && !leaderboard.cohortReady",
    "leaderboard?.cohortReady",
    "leaderboard.minimumCohort",
    "catalog.locationContract",
    "pack.role === 'OWNER'",
):
    require(packs, marker, f"Native Packs authority missing: {marker}")

packs_copy = normalized(packs)
for marker in (
    "Choose a coarse community, not a coordinate.",
    "The app never estimates or reconstructs a private local rank.",
    "Breadth in Human Skill and bounded Adventure variety count.",
    "Use a broad place people recognize.",
):
    require(packs_copy, marker, f"Native Packs boundary copy missing: {marker}")

for source_name, source in (("Community", feed), ("Packs", packs)):
    for forbidden in (
        ".sort(",
        "expo-location",
        "getCurrentPosition",
        "requestForegroundPermissions",
        "latitude",
        "longitude",
        "navigator.geolocation",
        "../api/daily-signals",
        "../api/pets",
        "../api/activities",
    ):
        if forbidden in source:
            raise SystemExit(f"Native {source_name} crossed a social authority boundary: {forbidden}")

if nav.count('name="Packs"') != 2:
    raise SystemExit("Packs must remain registered in both Guardian and Companion navigators")
require(nav, "Packs: undefined;", "Packs route type is missing")

pathway_match = re.search(
    r"export const SOCIAL_ADVENTURE_PATHWAYS = \[(.*?)\] as const;",
    server_policy,
    re.DOTALL,
)
if pathway_match is None:
    raise SystemExit("Unable to parse server Social Adventure pathway authority")
pathways = re.findall(r"'([A-Z_]+)'", pathway_match.group(1))
expected_pathways = ["MOVE", "EXPLORE", "ENRICH", "LEARN", "CONNECT", "RECOVER", "BOND"]
if pathways != expected_pathways:
    raise SystemExit(f"Server Social Adventure pathway set drifted: {pathways!r}")
if "CARE" in pathways:
    raise SystemExit("CARE must never enter Social Adventure competitive pathways")

require(
    server_policy,
    "export const LOCAL_LEAGUE_MINIMUM_COHORT = 5;",
    "Server local cohort privacy floor drifted",
)

for marker in (
    "pref.global_leaderboard_opt_in = TRUE",
    "u.visibility = 'PUBLIC'",
    "FROM public.blocked_users blocked",
    "cohortReady: false",
    "minimumCohort: LOCAL_LEAGUE_MINIMUM_COHORT",
    "locationContract: 'coarse-user-chosen-region-only'",
    "post.author_user_id = ${userId} OR post.visibility = 'PUBLIC'",
    "reactions: REACTION_TYPES.map",
):
    require(server_service, marker, f"Server Social Adventure privacy authority missing: {marker}")

for marker in (
    "does not create a second points economy",
    "Reactions are culture signals.",
    "private by default",
    "The client never derives, sorts, estimates, or repairs rank.",
    "coarse, user-chosen `regionKey`",
    "only renders entries when the server returns `cohortReady: true`",
    "The human gets the competition, collection, discovery, and community feedback.",
    "there is not yet a canonical Expedition API or receipt model",
):
    require(normalized(doc), marker, f"Native Social Adventure documentation boundary missing: {marker}")

print("Native Social Adventure source contract passed")
