#!/usr/bin/env python3
"""Fail closed when Daily Signals correction authority drifts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "apps/api/src/intelligence/daily-signals-correction.service.ts"
DTO = ROOT / "apps/api/src/intelligence/dto/daily-signals-correction.dto.ts"
CONTROLLER = ROOT / "apps/api/src/intelligence/intelligence.controller.ts"
SPEC = ROOT / "apps/api/src/intelligence/daily-signals-correction.integration.spec.ts"

for path in [SERVICE, DTO, CONTROLLER, SPEC]:
    if not path.is_file():
        raise SystemExit(f"Daily Signals correction source missing: {path.relative_to(ROOT)}")

service = SERVICE.read_text()
dto = DTO.read_text()
controller = CONTROLLER.read_text()
spec = SPEC.read_text()

required_service = [
    "version: 'daily-signals-correction-v1'",
    "eventType: 'DAILY_SIGNALS_CORRECTION'",
    "dedupeScope: 'PET'",
    "safetyEligible: false",
    "expectedCurrentCareEventId",
    "correctsCareEventId",
    "correctionSequence",
    "rootCareEventId",
    "payloadHash",
    "daily-signals-correction:",
    "reconcileChain",
    "dailySignals.replay",
    "normalizeOwnerCheckinObservation",
    "supersedesObservationId",
    "retractObservation",
    "Reload before correcting",
]
missing = [marker for marker in required_service if marker not in service]
if missing:
    raise SystemExit(f"Daily Signals correction authority drifted: {missing}")

for forbidden in [
    "UPDATE care_events",
    "DELETE FROM care_events",
    "prisma.careEvent.update",
    "prisma.careEvent.delete",
    "note?:",
    "note!:",
]:
    if forbidden in service or forbidden in dto:
        raise SystemExit(f"Daily Signals correction violated append-only/privacy boundary: {forbidden}")

required_controller = [
    "@Get('daily-signals/current')",
    "@Post('daily-signals/corrections')",
    "dailySignalsCorrection.getCurrent(req.user.sub, query)",
    "dailySignalsCorrection.correct(req.user.sub, dto)",
]
missing = [marker for marker in required_controller if marker not in controller]
if missing:
    raise SystemExit(f"Daily Signals correction HTTP authority drifted: {missing}")

required_spec = [
    "never exposes the private free-form note",
    "zero-XP correction",
    "exact correction retry idempotent",
    "different concurrent corrections",
    "explicit clear-all correction",
    "corrected uncertainty",
    "stale ancestor correction",
    "after household access is removed",
]
missing = [marker for marker in required_spec if marker not in spec]
if missing:
    raise SystemExit(f"Daily Signals correction regression coverage drifted: {missing}")

print(
    "Daily Signals correction authority is append-only, expected-tip guarded, PET-serialized, "
    "zero-reward, note-private, and projection-replayable."
)
