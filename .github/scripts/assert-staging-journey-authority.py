#!/usr/bin/env python3
"""Fail closed if deployed staging journey authority drifts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    target = ROOT / path
    if not target.is_file():
        raise SystemExit(f"required staging journey source missing: {path}")
    return target.read_text()


def require(path: str, *markers: str) -> None:
    text = read(path)
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise SystemExit(f"{path}: missing required staging journey markers: {missing}")


def reject(path: str, *markers: str) -> None:
    text = read(path)
    present = [marker for marker in markers if marker in text]
    if present:
        raise SystemExit(f"{path}: forbidden staging journey markers: {present}")


runner = ".github/scripts/verify-staging-user-journey.mjs"
workflow = ".github/workflows/deploy-staging.yml"

require(
    runner,
    "runStagingUserJourney",
    "createSyntheticIdentity",
    "registrationKey",
    "petCreationKey",
    "'/auth/register'",
    "'/auth/me'",
    "'/pets'",
    "'/users/me'",
    "deleted-session-rejected",
    "deleted-credentials-rejected",
    "finally",
    "cleanup attempted",
    "--self-test",
)
reject(
    runner,
    "DATABASE_URL",
    "PrismaClient",
    "console.log(identity",
    "console.log(token",
    "console.log(body",
    "response.text()",
)

require(
    workflow,
    "Run non-destructive live release smoke",
    "Run disposable staging user journey",
    "verify-staging-user-journey.mjs",
    "Retain Staging Release Receipt",
)

text = read(workflow)
smoke = text.index("Run non-destructive live release smoke")
journey = text.index("Run disposable staging user journey")
receipt = text.index("retain-staging-release-receipt:")
if not smoke < journey < receipt:
    raise SystemExit(
        "staging journey must run after live smoke and before staging receipt authority"
    )

print(
    "Staging journey authority is fail-closed: public HTTP lifecycle, owned-pet mutation, "
    "canonical account deletion, dead-session proof, privacy-safe logging, and failure cleanup are required."
)
