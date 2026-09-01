#!/usr/bin/env python3
"""Fail closed if legacy points regain runtime or client reward authority."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "apps/api/src/events/events.service.ts"
EVENTS_MODULE = ROOT / "apps/api/src/events/events.module.ts"
EVENTS_TEST = ROOT / "apps/api/src/events/events.reward-authority.spec.ts"
GAMIFICATION = ROOT / "apps/api/src/gamification/gamification.service.ts"
GAMIFICATION_MODULE = ROOT / "apps/api/src/gamification/gamification.module.ts"
GAMIFICATION_CONTROLLER = ROOT / "apps/api/src/gamification/gamification.controller.ts"
GAMIFICATION_TEST = ROOT / "apps/api/src/gamification/gamification.readonly.spec.ts"
CARE_EVENTS = ROOT / "apps/api/src/care-events/care-events.service.ts"
CARE_EVENTS_TEST = ROOT / "apps/api/src/care-events/care-events.integration.spec.ts"
SOCIAL = ROOT / "apps/api/src/social-adventure/social-adventure.service.ts"
PROFILE = ROOT / "apps/web/src/app/profile/page.tsx"
WEB_API = ROOT / "apps/web/src/lib/api.ts"
MOBILE_PROFILE = ROOT / "apps/mobile/src/screens/ProfileScreen.tsx"
MOBILE_GAMIFICATION = ROOT / "apps/mobile/src/api/gamification.ts"
DOC = ROOT / "docs/DOGOS_EVENTS_REWARD_AUTHORITY.md"
WORKFLOW = ROOT / ".github/workflows/dogos-events-reward-authority-ci.yml"
LEGACY_WEB_STORE = ROOT / "apps/web/src/lib/stores/gamification-store.ts"
RETIRED_DTOS = [
    ROOT / "apps/api/src/gamification/dto/award-points.dto.ts",
    ROOT / "apps/api/src/gamification/dto/award-badge.dto.ts",
    ROOT / "apps/api/src/gamification/dto/update-streak.dto.ts",
]

for path in [
    EVENTS,
    EVENTS_MODULE,
    EVENTS_TEST,
    GAMIFICATION,
    GAMIFICATION_MODULE,
    GAMIFICATION_CONTROLLER,
    GAMIFICATION_TEST,
    CARE_EVENTS,
    CARE_EVENTS_TEST,
    SOCIAL,
    PROFILE,
    WEB_API,
    MOBILE_PROFILE,
    DOC,
    WORKFLOW,
]:
    if not path.is_file():
        raise SystemExit(f"required reward-authority source missing: {path.relative_to(ROOT)}")

for retired_path, label in [
    (LEGACY_WEB_STORE, "unreachable client-side legacy gamification store"),
    (MOBILE_GAMIFICATION, "phantom mobile gamification API client"),
]:
    if retired_path.exists():
        raise SystemExit(f"{label} must remain retired")
for path in RETIRED_DTOS:
    if path.exists():
        raise SystemExit(f"retired legacy mutation DTO returned: {path.relative_to(ROOT)}")

events = EVENTS.read_text()
for marker in [
    "eventRSVP.updateMany({",
    "checkedInAt: null",
    "eventFeedback.upsert({",
    "tags: dto.tags || []",
    "Already checked in. Attendance is unchanged.",
    "Feedback saved. Thanks for helping the community learn about this event.",
]:
    if marker not in events:
        raise SystemExit(f"event participation authority marker missing: {marker}")
for forbidden in [
    "GamificationService",
    "awardPoints",
    "pointsAwarded",
    "event_attended",
    "event_feedback",
    "You earned",
    "totalPoints",
    "pointTransaction",
    "badgeAward",
    "weeklyStreak",
]:
    if forbidden in events:
        raise SystemExit(f"Events regained legacy reward authority: {forbidden}")

if "GamificationModule" in EVENTS_MODULE.read_text():
    raise SystemExit("EventsModule must not depend on legacy GamificationModule")

events_test = EVENTS_TEST.read_text()
for marker in [
    "one conditional transition and no reward response",
    "repeated or concurrent-loser check-in as an acknowledged no-op",
    "uses one composite-key upsert for feedback and exposes no reward semantics",
    "preserves feedback replacement semantics when optional tags are omitted",
    "not.toHaveProperty('pointsAwarded')",
]:
    if marker not in events_test:
        raise SystemExit(f"event reward-authority defining test missing: {marker}")

gamification = GAMIFICATION.read_text()
for marker in [
    "async getUserPoints(",
    "async getUserBadges(",
    "async getUserStreak(",
    "return { ...streak, currentWeek: 0 }",
]:
    if marker not in gamification:
        raise SystemExit(f"legacy read-only compatibility marker missing: {marker}")
for forbidden in [
    "async getPointTransactions(",
    "async getLeaderboard(",
    "awardPoints(",
    "awardBadge(",
    "updateStreak(",
    ".create({",
    ".update({",
    ".updateMany({",
    ".upsert({",
    ".delete({",
    ".deleteMany({",
    "$executeRaw",
]:
    if forbidden in gamification:
        raise SystemExit(f"legacy compatibility service contains retired or mutation authority: {forbidden}")

module = GAMIFICATION_MODULE.read_text()
if "exports:" in module:
    raise SystemExit("GamificationModule must not export the legacy compatibility service")

controller = GAMIFICATION_CONTROLLER.read_text()
for marker in ["@Get('me/summary')", "deprecated: true", "replacement: '/adventure/me'"]:
    if marker not in controller:
        raise SystemExit(f"legacy compatibility controller marker missing: {marker}")
for forbidden in ["@Post(", "@Put(", "@Patch(", "@Delete("]:
    if forbidden in controller:
        raise SystemExit(f"legacy gamification controller regained mutation endpoint: {forbidden}")

gamification_test = GAMIFICATION_TEST.read_text()
for marker in [
    "normalizes an expired streak in memory without mutating legacy storage",
    "returns historical totals without writing them",
    "returns historical badges without creating new awards",
]:
    if marker not in gamification_test:
        raise SystemExit(f"read-only legacy compatibility test missing: {marker}")

care_events = CARE_EVENTS.read_text()
for marker in ["INSERT INTO reward_ledger", "bondXp: reward.bondXp", "rewardCareEvent(input"]:
    if marker not in care_events:
        raise SystemExit(f"canonical Bond XP authority marker missing: {marker}")
if re.search(r"totalPoints\s*:\s*\{\s*increment", care_events, re.MULTILINE):
    raise SystemExit("canonical Bond XP must not mirror into retired users.totalPoints")
if "Keep the legacy aggregate synchronized" in care_events:
    raise SystemExit("legacy total-points synchronization comment/runtime returned")

care_events_test = CARE_EVENTS_TEST.read_text()
for marker in [
    "without mutating legacy points",
    "expect(newlyIssued[0]?.bondXp).toBeGreaterThan(0)",
    "expect(user.totalPoints).toBe(0)",
]:
    if marker not in care_events_test:
        raise SystemExit(f"Bond XP / legacy-points separation test missing: {marker}")

# No maintained API domain may call the retired mutation surface or inject the legacy service.
for path in (ROOT / "apps/api/src").rglob("*.ts"):
    if path.is_relative_to(ROOT / "apps/api/src/gamification"):
        continue
    text = path.read_text()
    for forbidden in ["awardPoints(", "awardBadge(", "updateStreak(", "GamificationService"]:
        if forbidden in text:
            raise SystemExit(
                f"retired legacy gamification authority {forbidden!r} found in {path.relative_to(ROOT)}"
            )

# Universal legacy total-point increment authority is retired. Reads/selects remain compatibility data.
for path in (ROOT / "apps/api/src").rglob("*.ts"):
    text = path.read_text()
    if re.search(r"totalPoints\s*:\s*\{\s*increment", text, re.MULTILINE):
        raise SystemExit(f"legacy totalPoints increment returned in {path.relative_to(ROOT)}")

social = SOCIAL.read_text()
if "totalPoints" in social or "awardPoints" in social or "GamificationService" in social:
    raise SystemExit("Social Adventure must remain independent of legacy points")

profile = PROFILE.read_text()
if not re.search(r"label:\s*['\"]Legacy points['\"]", profile):
    raise SystemExit("Web profile must label frozen historical total as Legacy points")
if re.search(r"label:\s*['\"]Points['\"]", profile):
    raise SystemExit("Web profile must not present legacy aggregate as active Points")

web_api = WEB_API.read_text()
if "export const gamificationApi" in web_api:
    raise SystemExit("Web generic API must not expose phantom legacy gamification authority")

mobile_profile = MOBILE_PROFILE.read_text()
for marker in [
    "petsApi.getPets(user.id)",
    "navigation.navigate('Pets')",
    "navigation.navigate('Goals')",
    "navigation.navigate('Events')",
    "navigation.navigate('Library')",
    "Pets are temporarily unavailable",
]:
    if marker not in mobile_profile:
        raise SystemExit(f"mobile profile maintained-boundary marker missing: {marker}")
for forbidden in [
    "gamificationApi",
    "navigation.navigate('Settings')",
    "navigation.navigate('EditProfile')",
    "navigation.navigate('PetsList')",
    "navigation.navigate('PetDetail')",
    "navigation.navigate('Activities')",
    "navigation.navigate('Leaderboard')",
    ">Points<",
    ">Level<",
    ">Rank<",
    ">Badges<",
]:
    if forbidden in mobile_profile:
        raise SystemExit(f"mobile profile retained phantom reward/navigation authority: {forbidden}")

# Client source may use the deprecated read-only /gamification/me/summary endpoint, but no
# retired mutation, leaderboard, stats, badge, or arbitrary profile route may remain.
phantom_client_routes = [
    "/gamification/points",
    "/gamification/leaderboard",
    "/gamification/stats",
    "/gamification/badges",
    "/gamification/profile/",
]
for client_root in [ROOT / "apps/web/src", ROOT / "apps/mobile/src"]:
    for path in client_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".ts", ".tsx", ".js", ".jsx"}:
            continue
        text = path.read_text()
        for forbidden in phantom_client_routes:
            if forbidden in text:
                raise SystemExit(
                    f"phantom legacy gamification route {forbidden!r} found in {path.relative_to(ROOT)}"
                )

doc = DOC.read_text()
for marker in [
    "acknowledgement-only participation signals",
    "Legacy total-points freeze",
    "Bond XP is no longer mirrored into `users.totalPoints`",
    "The Web profile labels this field **Legacy points**",
    "Historical rows are not deleted or migrated by this release",
    "phantom Web and Mobile gamification clients",
]:
    if marker not in doc:
        raise SystemExit(f"reward authority documentation marker missing: {marker}")

workflow = WORKFLOW.read_text()
for marker in [
    ".github/scripts/assert-events-reward-authority.py",
    "apps/api/src/events/**",
    "apps/api/src/gamification/**",
    "apps/api/src/care-events/**",
    "apps/api/src/social-adventure/**",
    "apps/web/src/app/profile/page.tsx",
    "apps/web/src/lib/api.ts",
    "apps/mobile/src/api/gamification.ts",
    "apps/mobile/src/screens/ProfileScreen.tsx",
    "docs/DOGOS_EVENTS_REWARD_AUTHORITY.md",
    "events.reward-authority.spec.ts",
    "gamification.readonly.spec.ts",
    "care-events.integration.spec.ts",
    "python .github/scripts/assert-events-reward-authority.py",
    "pnpm --filter @woof/mobile type-check",
]:
    if marker not in workflow:
        raise SystemExit(f"reward-authority CI ownership marker missing: {marker}")

print(
    "Event reward authority is explicit: participation is retry-safe and acknowledgement-only, "
    "legacy gamification is read-only, totalPoints is frozen, client ghosts are retired, "
    "and Bond/Social economies remain separate."
)
