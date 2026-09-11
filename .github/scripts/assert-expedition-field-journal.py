#!/usr/bin/env python3
"""Fail closed if Expedition Field Journal drifts into an achievement or score system."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SERVICE = ROOT / "apps/api/src/expeditions/expedition-journal.service.ts"
CONTROLLER = ROOT / "apps/api/src/expeditions/expeditions.controller.ts"
MODULE = ROOT / "apps/api/src/expeditions/expeditions.module.ts"
INTEGRATION = ROOT / "apps/api/src/expeditions/expedition-journal.integration.spec.ts"
MOBILE_API = ROOT / "apps/mobile/src/api/expeditions.ts"
SCREEN = ROOT / "apps/mobile/src/screens/ExpeditionScreen.tsx"
WORLD = ROOT / "apps/mobile/src/components/community/ExpeditionWorldView.tsx"
JOURNAL_VIEW = ROOT / "apps/mobile/src/components/community/ExpeditionFieldJournalView.tsx"
MIGRATION = (
    ROOT
    / "packages/database/prisma/migrations/20260909181500_add_expedition_authority_v1/migration.sql"
)
DOC = ROOT / "docs/NATIVE_EXPEDITION_FIELD_JOURNAL_V1.md"


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
        SERVICE,
        CONTROLLER,
        MODULE,
        INTEGRATION,
        MOBILE_API,
        SCREEN,
        WORLD,
        JOURNAL_VIEW,
        MIGRATION,
        DOC,
    ]
    for path in required:
        if not path.is_file():
            fail(f"missing required Expedition Journal file: {path.relative_to(ROOT)}")

    service = SERVICE.read_text()
    controller = CONTROLLER.read_text()
    module = MODULE.read_text()
    integration = INTEGRATION.read_text()
    mobile_api = MOBILE_API.read_text()
    screen = SCREEN.read_text()
    world = WORLD.read_text()
    journal_view = JOURNAL_VIEW.read_text()
    migration = MIGRATION.read_text()
    doc = DOC.read_text()

    # The Journal is a read projection over canonical receipt authority. The existing
    # Expedition service remains the only current-season materializer.
    for marker in [
        "JOURNAL_MAX_PARTICIPATED_SEASONS = 26",
        "await this.expeditions.getGlobal(userId)",
        "receipt.expedition_key = ${EXPEDITION_KEY}",
        "receipt.expedition_version = ${EXPEDITION_VERSION}",
        "receipt.policy_version = ${EXPEDITION_POLICY_VERSION}",
        "receipt.scope = 'GLOBAL'",
        "receipt.pack_id IS NULL",
        "receipt.user_id = ${userId}",
        "receipt.season_key <= ${current.key}",
        "GROUP BY receipt.season_key, receipt.objective_key",
        "LIMIT ${JOURNAL_MAX_PARTICIPATED_SEASONS}",
        "startsAt.getUTCDay() !== 1",
        "landmarks.add(objective.key)",
        "landmarks: EXPEDITION_OBJECTIVES.filter",
        "kind: 'RECENT_PARTICIPATED_SEASONS' as const",
        "'landmark-presence-not-volume'",
        "'no-completion-or-rarity'",
        "'no-rank-or-streak'",
    ]:
        require(service, marker, "server Expedition Journal")

    for forbidden in [
        "INSERT INTO",
        "UPDATE dogos_social",
        "DELETE FROM",
        "scope = 'PACK'",
        "pack_id = ${",
        "leaderboard",
        "socialAdventure",
        "score:",
        "rank:",
        "target:",
        "rarity:",
        "completed:",
        "streak:",
    ]:
        reject(service, forbidden, "server Expedition Journal")

    # Route is authenticated by the existing controller guard and GET-only.
    require(controller, "@UseGuards(JwtAuthGuard)", "Expedition controller")
    require(controller, "@Get('journal')", "Expedition controller")
    require(controller, "return this.journal.getMine(req.user.sub)", "Expedition controller")
    require_count(controller, "@Get('journal')", 1, "Expedition controller")
    for marker in ["@Post('journal')", "@Put('journal')", "@Patch('journal')", "@Delete('journal')"]:
        reject(controller, marker, "Expedition controller")

    require(module, "providers: [ExpeditionsService, ExpeditionJournalService]", "Expedition module")

    # The integration contract intentionally injects hostile historical rows and proves
    # volume, another user, Pack scope, future weeks, and non-Monday weeks do not leak.
    for marker in [
        "insertCareEvent(userId, evidenceAt, 0)",
        "insertCareEvent(userId, new Date(evidenceAt.getTime() + 1000), 1)",
        "userId: otherUserId",
        "scope: 'PACK'",
        "suffix: 'future'",
        "suffix: 'non-monday'",
        "maxSeasons: 26",
        "{ key: 'SNIFF_EXPLORE', title: 'Sniff & Explore' }",
        "{ key: 'READ_THE_ROOM', title: 'Read the Room' }",
        "{ key: 'RECOVERY_COUNTS', title: 'Recovery Counts' }",
        "expect(Object.keys(current ?? {}).sort()).toEqual(['landmarks', 'season', 'state'])",
        "expect(Object.keys(current?.landmarks[0] ?? {}).sort()).toEqual(['key', 'title'])",
    ]:
        require(integration, marker, "Expedition Journal integration test")

    # Mobile history type is deliberately narrow. The journal has no count, score, rank,
    # target, completion, or Pack fields that presentation could repurpose later.
    journal_type_start = mobile_api.find("export type ExpeditionJournalEntry")
    journal_type_end = mobile_api.find("export const expeditionApi")
    if journal_type_start < 0 or journal_type_end <= journal_type_start:
        fail("mobile Expedition Journal types are missing or malformed")
    journal_types = mobile_api[journal_type_start:journal_type_end]

    for marker in [
        "state: 'ACTIVE' | 'PAST'",
        "landmarks: {\n    key: ExpeditionObjectiveKey;\n    title: string;\n  }[];",
        "kind: 'RECENT_PARTICIPATED_SEASONS'",
        "maxSeasons: number",
        "scope: 'GLOBAL'",
    ]:
        require(journal_types, marker, "mobile Expedition Journal types")

    for forbidden in [
        "count:",
        "total:",
        "contributors:",
        "myContribution:",
        "score:",
        "rank:",
        "target:",
        "rarity:",
        "completed:",
        "streak:",
        "pack:",
    ]:
        reject(journal_types, forbidden, "mobile Expedition Journal types")

    require(
        mobile_api,
        "journal: () => apiClient.get<ExpeditionJournal>('/expeditions/journal')",
        "mobile Expedition API",
    )
    require_count(mobile_api, "'/expeditions/journal'", 1, "mobile Expedition API")
    for forbidden in [
        "apiClient.post<ExpeditionJournal>",
        "apiClient.put<ExpeditionJournal>",
        "apiClient.patch<ExpeditionJournal>",
        "apiClient.delete<ExpeditionJournal>",
    ]:
        reject(mobile_api, forbidden, "mobile Expedition API")

    # Journal is independently degradable. Its failure cannot poison a healthy live world,
    # and no current-world or Social Adventure state is used to reconstruct history.
    for marker in [
        "type ExpeditionJournal",
        "expeditionApi.journal()",
        "Promise.allSettled([",
        "journalResult.status === 'fulfilled' && journalResult.value.scope === 'GLOBAL'",
        "journalRef.current = journalResult.value",
        "setJournalError(null)",
        "journalError={journalError}",
        "setWorldError(",
        "setJournalError(",
        "leave history blank rather than guess",
    ]:
        require(screen, marker, "native Expedition screen")
    reject(screen, "unavailableWorld.push('field journal')", "native Expedition screen")
    reject(screen, "unavailable.push('field journal')", "native Expedition screen")

    require(
        world,
        "<ExpeditionFieldJournalView journal={props.journal} error={props.journalError} />",
        "native Expedition world",
    )
    require(world, "timeZone: 'UTC'", "native Expedition world")

    # The scrapbook only renders returned landmarks. There are no missing-slot placeholders,
    # page fullness calculations, or local transformations into achievement semantics.
    for marker in [
        "entry.landmarks.map((landmark)",
        "does not make a bigger stamp",
        "There is nothing you need to fill.",
        "Blank space is part of the memory.",
        "Nothing is overdue and there is nothing to catch up on.",
        "older pages are not presented as non-participation.",
        "timeZone: 'UTC'",
        "Your shared world is still available.",
        "Showing your last verified pages.",
    ]:
        require(journal_view + screen, marker, "native Expedition Field Journal")

    for forbidden in [
        "entry.landmarks.length",
        "journal.entries.length /",
        "landmarks.length /",
        "Math.min(",
        "Math.max(",
        "* 100",
        "ProgressBar",
        "percentComplete",
        "completionRate",
        "unlockThreshold",
        "rarityLevel",
        ".score",
        ".rank",
        ".target",
        ".completed",
        ".streak",
        "missingLandmarks",
        "emptyStamp",
        "lockedStamp",
    ]:
        reject(journal_view, forbidden, "native Expedition Field Journal")

    require_count(journal_view, "journal.coverage.maxSeasons", 1, "native Expedition Field Journal")

    # Existing receipt persistence remains immutable and indexed for user-season reads.
    for marker in [
        "CREATE INDEX IF NOT EXISTS expedition_receipt_user_season_idx",
        "ON dogos_social.expedition_receipts (user_id, season_key, authorized_at DESC, id)",
        "CREATE TRIGGER expedition_receipts_immutable_after_issue",
        "RAISE EXCEPTION 'expedition contribution receipts are immutable'",
    ]:
        require(migration, marker, "Expedition receipt migration")

    for marker in [
        "A field note says **this happened**.",
        "Presence, never volume",
        "It does not render empty slots for the other landmarks.",
        "No state called `COMPLETE`, `FAILED`, `MISSED`",
        "Field Journal v1 is personal and Global-only.",
        "26 most recent participated seasons",
        "missing older pages are not presented as non-participation",
        "Season dates are formatted in UTC",
        "Field Journal v1 is **not** Story persistence",
        "how do we make real weeks worth remembering?",
    ]:
        require(doc, marker, "Expedition Field Journal documentation")

    print("Expedition Field Journal authority contract OK")


if __name__ == "__main__":
    main()
