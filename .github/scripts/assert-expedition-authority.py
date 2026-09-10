import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"Missing {label}: {needle}")


def require_regex(text: str, pattern: str, label: str) -> None:
    if re.search(pattern, text, re.DOTALL) is None:
        raise SystemExit(f"Missing {label}: pattern {pattern!r}")


def require_count(text: str, needle: str, minimum: int, label: str) -> None:
    count = text.count(needle)
    if count < minimum:
        raise SystemExit(f"Missing {label}: expected >= {minimum} occurrences of {needle!r}, found {count}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise SystemExit(f"Forbidden {label}: {needle}")


controller = read("apps/api/src/expeditions/expeditions.controller.ts")
service = read("apps/api/src/expeditions/expeditions.service.ts")
policy = read("apps/api/src/expeditions/expeditions.policy.ts")
module = read("apps/api/src/expeditions/expeditions.module.ts")
adventure_module = read("apps/api/src/adventure/adventure.module.ts")
legacy_service = read("apps/api/src/adventure/pack-challenges.service.ts")
pack_access = read("apps/api/src/social-adventure/pack-access.service.ts")
social_module = read("apps/api/src/social-adventure/social-adventure.module.ts")
migration = read(
    "packages/database/prisma/migrations/20260909181500_add_expedition_authority_v1/migration.sql"
)
docs = read("docs/EXPEDITION_AUTHORITY_V1.md")

# API surface: authority v1 is read-only. There is no client progress mutation.
for needle in [
    "@Controller('expeditions')",
    "@UseGuards(JwtAuthGuard)",
    "@Get('global')",
    "@Get('packs/:packId')",
]:
    require(controller, needle, "Expedition read endpoint")
for needle in ["@Post(", "@Put(", "@Patch(", "@Delete(", "@Body("]:
    forbid(controller, needle, "client contribution mutation surface")

# Module graph must register the controller and make ExpeditionsService available to
# the deprecated /pack/challenges adapter through AdventureModule.
require(module, "controllers: [ExpeditionsController]", "Expedition controller registration")
require(module, "exports: [ExpeditionsService]", "Expedition service export")
require(adventure_module, "ExpeditionsModule", "Adventure module Expedition import")
require(
    adventure_module,
    "imports: [InsightsModule, CareEventsModule, HouseholdsModule, ExpeditionsModule]",
    "Adventure module wiring",
)

# Pack authorization is one shared server authority, and Pack Expedition reads demand
# ACTIVE membership rather than merely a public Pack or cached client flag.
require(pack_access, "member.status = 'ACTIVE'", "ACTIVE Pack membership predicate")
require(pack_access, "requireActiveMembership", "Pack membership authority")
require(
    pack_access,
    "if (!pack || !pack.viewerJoined)",
    "non-enumerable Pack membership rejection",
)
require_regex(
    social_module,
    r"providers\s*:\s*\[[^\]]*\bPackAccessService\b[^\]]*\]",
    "shared Pack authority provider",
)
require_regex(
    social_module,
    r"exports\s*:\s*\[[^\]]*\bPackAccessService\b[^\]]*\]",
    "shared Pack authority export",
)
require(
    service,
    "this.packAccess.requireActiveMembership(userId, packId)",
    "Pack Expedition membership gate",
)
require(
    service,
    "event.occurred_at >= GREATEST(${season.startsAt}, member.joined_at)",
    "post-join CareEvent boundary",
)
require(
    service,
    "attempt.completed_at >= GREATEST(${season.startsAt}, member.joined_at)",
    "post-join Human Skill boundary",
)

# Policy is breadth-first and explicitly excludes CARE.
require(
    policy,
    "EXPEDITION_POLICY_VERSION = 'expedition-authority-v1'",
    "policy version",
)
require(policy, "EXPEDITION_KEY = 'shared-world'", "Expedition key")
require(policy, "EXPEDITION_VERSION = 'v1'", "Expedition version")
require(
    policy,
    "EXPEDITION_ELIGIBLE_PATHWAYS = ['EXPLORE', 'ENRICH', 'RECOVER']",
    "eligible pathway allowlist",
)
for key in ["MAKE_IT_EASIER", "CATCH_THE_GOOD", "PAIRING_LAB", "MARKER_TIMING"]:
    require(policy, f"'{key}'", f"Human Skill category {key}")
for objective in ["SNIFF_EXPLORE", "RECOVERY_COUNTS", "READ_THE_ROOM"]:
    require(policy, f"key: '{objective}'", f"objective {objective}")
for cap in [
    "perContributorCap: 4",
    "perContributorCap: 2",
    "perCategoryCap: 2",
    "perCategoryCap: 1",
]:
    require(policy, cap, "bounded contribution policy")

# Canonical evidence, incremental reconciliation, deterministic bounded ranking, and
# idempotent receipt issuance. Both Global and Pack paths must preserve already-issued
# allowances so neither repeated reads nor Pack rejoin can reset a seasonal cap.
require(service, "event.source = 'QUEST_ENGINE'", "canonical Adventure source")
require(service, "event.event_type LIKE 'QUEST_%'", "canonical Adventure event family")
require(
    service,
    "event.pathway IN ('EXPLORE', 'ENRICH', 'RECOVER')",
    "eligible Adventure pathways",
)
for challenge in [
    "'MAKE_IT_EASIER'",
    "'CATCH_THE_GOOD'",
    "'PAIRING_LAB'",
    "'MARKER_TIMING'",
]:
    require(service, challenge, "canonical Human Skill allowlist")
require_count(
    service,
    "COALESCE(issued.issued_count, 0)",
    4,
    "issued-cap accounting across Global and Pack materializers",
)
require_count(
    service,
    "category_rank + issued_count <= 2",
    2,
    "CareEvent cap across Global and Pack materializers",
)
require_count(
    service,
    "category_rank + issued_count <= 1",
    2,
    "Human Skill cap across Global and Pack materializers",
)
require_count(
    service,
    "AND NOT EXISTS (",
    4,
    "already-receipted source exclusion across materializers",
)
require(service, "ON CONFLICT DO NOTHING", "idempotent receipt materialization")
require(service, "target: null", "uncalibrated target contract")
require(service, "status: 'CALIBRATING'", "uncalibrated status contract")
for forbidden in [
    "context->>'activityMinutes'",
    "context->>'distance'",
    "context->>'mileage'",
    "context->>'calories'",
    "likes_count",
    "comments_count",
    ".score AS",
    "attempt.score",
]:
    forbid(service, forbidden, "raw performance/popularity/practice-score scoring input")

# The deprecated route is a compatibility adapter over the Global receipt projection,
# not a second raw care_events aggregate.
require(legacy_service, "getLegacyGlobalChallenges", "legacy Global Expedition adapter")
forbid(legacy_service, "FROM care_events", "legacy raw cooperative aggregate")
forbid(legacy_service, "$queryRaw", "legacy second truth engine")

# Database receipts carry scope/provenance and cannot be rewritten after issuance.
for needle in [
    "CREATE TABLE IF NOT EXISTS dogos_social.expedition_receipts",
    "scope IN ('GLOBAL', 'PACK')",
    "scope = 'GLOBAL' AND pack_id IS NULL",
    "scope = 'PACK' AND pack_id IS NOT NULL",
    "source_type IN ('CARE_EVENT', 'HUMAN_SKILL_ATTEMPT')",
    "pathway IN ('EXPLORE', 'ENRICH', 'RECOVER')",
    "CREATE UNIQUE INDEX IF NOT EXISTS expedition_receipt_evidence_unique",
    "policy_version",
    "reject_expedition_receipt_update",
    "BEFORE UPDATE ON dogos_social.expedition_receipts",
]:
    require(migration, needle, "receipt database authority")
forbid(migration, "'CARE'", "CARE receipt pathway")

# Product contract remains explicit about game boundaries and calibration.
for needle in [
    "The human gets the cooperative game. The dog keeps the right to have an ordinary day.",
    "No client contribution endpoint",
    "`CARE` is structurally excluded",
    "Practice score magnitude",
    "target: null",
    "status: CALIBRATING",
    "There is one cooperative truth engine.",
    "free-form Pack locality or device location",
]:
    require(docs, needle, "documented Expedition authority boundary")

print("Expedition Authority v1 source contract: PASS")
