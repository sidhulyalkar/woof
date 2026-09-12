#!/usr/bin/env python3
"""Qualify staging and production promotion authority without deployment credentials."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / ".github/workflows/deploy-staging.yml"
PRODUCTION = ROOT / ".github/workflows/deploy-production.yml"
RECEIPT = ROOT / ".github/scripts/write-release-receipt.mjs"


def read(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"required release authority source missing: {path.relative_to(ROOT)}")
    return path.read_text()


def require(text: str, label: str, *markers: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise SystemExit(f"{label}: missing required release-authority markers: {missing}")


def reject(text: str, label: str, *markers: str) -> None:
    present = [marker for marker in markers if marker in text]
    if present:
        raise SystemExit(f"{label}: forbidden release-authority markers present: {present}")


staging = read(STAGING)
production = read(PRODUCTION)
receipt = read(RECEIPT)

require(
    staging,
    "deploy-staging.yml",
    "branches: [main]",
    "workflow_dispatch:",
    "REQUESTED_SHA: ${{ github.sha }}",
    "Validate Staging Release Candidate",
    "git merge-base --is-ancestor \"${RELEASE_SHA}\" origin/main",
    "ref: ${{ env.RELEASE_SHA }}",
    '--build-arg WOOF_RELEASE_SHA="${RELEASE_SHA}"',
    "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ env.RELEASE_SHA }}",
    "Retain Staging Release Receipt",
    "staging-release-${{ env.RELEASE_SHA }}",
    "retention-days: 30",
)
reject(staging, "deploy-staging.yml", "branches: [develop]")

require(
    production,
    "deploy-production.yml",
    "workflow_dispatch:",
    "release_sha:",
    "required: true",
    "actions: read",
    "Validate Production Release Candidate",
    "git merge-base --is-ancestor \"${RELEASE_SHA}\" origin/main",
    "Verify exact staging qualification",
    "actions/workflows/deploy-staging.yml/runs",
    '.conclusion == "success"',
    "ref: ${{ env.RELEASE_SHA }}",
    '--build-arg WOOF_RELEASE_SHA="${RELEASE_SHA}"',
    "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ env.RELEASE_SHA }}",
    "environment: production",
    "Retain Production Release Receipt",
    "production-release-${{ env.RELEASE_SHA }}",
    "retention-days: 90",
)
reject(
    production,
    "deploy-production.yml",
    "\n  push:\n",
    "branches: [main]",
    'WOOF_RELEASE_SHA="${GITHUB_SHA}"',
    "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ github.sha }}",
)

require(
    receipt,
    "write-release-receipt.mjs",
    "SHA_PATTERN",
    "ALLOWED_ENVIRONMENTS",
    "mainAncestryVerified: true",
    "apiReleaseIdentityVerified: true",
    "webReleaseIdentityVerified: true",
    "webApiOriginVerified: true",
    "production receipt requires successful exact-SHA staging qualification",
    "--self-test",
)
reject(receipt, "write-release-receipt.mjs", "process.env.FLY_API_TOKEN", "process.env.VERCEL_TOKEN")

print(
    "Release promotion authority is explicit: main stages automatically, production is manual exact-SHA promotion, "
    "successful staging is required, and privacy-safe receipts are retained."
)
