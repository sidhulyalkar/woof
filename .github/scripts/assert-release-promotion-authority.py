#!/usr/bin/env python3
"""Qualify staging and production promotion authority without deployment credentials."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / ".github/workflows/deploy-staging.yml"
PRODUCTION = ROOT / ".github/workflows/deploy-production.yml"
RELEASE_CI = ROOT / ".github/workflows/release-promotion-authority-ci.yml"
CHECK_VERIFIER = ROOT / ".github/scripts/verify-main-release-checks.py"
LIVE_VERIFIER = ROOT / ".github/scripts/verify-live-release.mjs"
RECEIPT = ROOT / ".github/scripts/write-release-receipt.mjs"
DEPLOYMENT_GUIDE = ROOT / "DEPLOYMENT_GUIDE.md"


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
release_ci = read(RELEASE_CI)
check_verifier = read(CHECK_VERIFIER)
live_verifier = read(LIVE_VERIFIER)
receipt = read(RECEIPT)
deployment_guide = read(DEPLOYMENT_GUIDE)

require(
    staging,
    "deploy-staging.yml",
    "branches: [main]",
    "workflow_dispatch:",
    "actions: read",
    "REQUESTED_SHA: ${{ github.sha }}",
    "Validate Staging Release Candidate",
    "git merge-base --is-ancestor \"${RELEASE_SHA}\" origin/main",
    "Require exact-SHA canonical main release checks",
    "python3 .github/scripts/verify-main-release-checks.py",
    "ref: ${{ env.RELEASE_SHA }}",
    '--build-arg WOOF_RELEASE_SHA="${RELEASE_SHA}"',
    "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ env.RELEASE_SHA }}",
    "STABLE_WEB_ORIGIN: ${{ vars.WEB_ORIGIN }}",
    "vercel pull --yes --environment=production",
    "vercel build --prod",
    "vercel deploy --prebuilt --prod",
    "Verify stable staging Web origin release and API integration",
    "Run non-destructive live release smoke",
    "node .github/scripts/verify-live-release.mjs",
    "CANONICAL_RELEASE_CHECKS_VERIFIED: 'true'",
    "LIVE_BLACK_BOX_VERIFIED: 'true'",
    "Retain Staging Release Receipt",
    "staging-release-${{ env.RELEASE_SHA }}",
    "retention-days: 30",
)
reject(staging, "deploy-staging.yml", "branches: [develop]", "--environment=preview")

require(
    production,
    "deploy-production.yml",
    "workflow_dispatch:",
    "release_sha:",
    "required: true",
    "actions: read",
    "Validate Production Release Candidate",
    "git merge-base --is-ancestor \"${RELEASE_SHA}\" origin/main",
    "Require exact-SHA canonical main release checks",
    "python3 .github/scripts/verify-main-release-checks.py",
    "Verify exact staging qualification",
    "actions/workflows/deploy-staging.yml/runs",
    '.conclusion == "success"',
    "ref: ${{ env.RELEASE_SHA }}",
    '--build-arg WOOF_RELEASE_SHA="${RELEASE_SHA}"',
    "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ env.RELEASE_SHA }}",
    "STABLE_WEB_ORIGIN: ${{ vars.WEB_ORIGIN }}",
    "Verify stable production Web origin release and API integration",
    "Run non-destructive live release smoke",
    "node .github/scripts/verify-live-release.mjs",
    "environment: production",
    "CANONICAL_RELEASE_CHECKS_VERIFIED: 'true'",
    "LIVE_BLACK_BOX_VERIFIED: 'true'",
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
    release_ci,
    "release-promotion-authority-ci.yml",
    "pull_request:",
    "branches: [main]",
    "push:",
    "Release Promotion Authority",
    "verify-main-release-checks.py --self-test",
    "verify-live-release.mjs --self-test",
)
reject(release_ci, "release-promotion-authority-ci.yml", "paths:")

for workflow in [
    '"ci.yml"',
    '"security-baseline-ci.yml"',
    '"codeql.yml"',
    '"release-promotion-authority-ci.yml"',
]:
    require(check_verifier, "verify-main-release-checks.py", workflow)
require(
    check_verifier,
    "verify-main-release-checks.py",
    'run.get("head_sha") != release_sha',
    'run.get("event") != "push"',
    'success="success" in states',
    "--self-test",
)

require(
    live_verifier,
    "verify-live-release.mjs",
    "ops/health/${endpoint}",
    "auth/me",
    "x-content-type-options",
    "access-control-allow-origin",
    "access-control-allow-credentials",
    "production-shaped API documentation is unexpectedly reachable",
    "--self-test",
)
reject(live_verifier, "verify-live-release.mjs", "Authorization: Bearer", "email", "petId")

require(
    receipt,
    "write-release-receipt.mjs",
    "schemaVersion: 2",
    "SHA_PATTERN",
    "ALLOWED_ENVIRONMENTS",
    "mainAncestryVerified: true",
    "canonicalReleaseChecksVerified",
    "apiReleaseIdentityVerified: true",
    "webReleaseIdentityVerified: true",
    "webApiOriginVerified: true",
    "liveBlackBoxVerified",
    "webDeploymentUrl",
    "production receipt requires successful exact-SHA staging qualification",
    "--self-test",
)
reject(receipt, "write-release-receipt.mjs", "process.env.FLY_API_TOKEN", "process.env.VERCEL_TOKEN")

require(
    deployment_guide,
    "DEPLOYMENT_GUIDE.md",
    "Pre-public-beta deployment authority",
    "woof-api-staging",
    "woof-api-prod",
    "separate Vercel projects",
    "WEB_ORIGIN",
    "backup and restore rehearsal",
    "TestFlight",
    "repository-qualified ≠ deployed ≠ device-qualified ≠ pilot-validated",
)
reject(
    deployment_guide,
    "DEPLOYMENT_GUIDE.md",
    "Ready for deployment ✅",
    "pnpm --filter @woof/database db:seed",
    "API_DOCS_ENABLED=true",
    "PWA Configuration",
)

print(
    "Release promotion authority is fail-closed: exact main SHA, canonical release checks, isolated stable Web origin, "
    "successful staging, live black-box proof, and privacy-safe receipts are required before production is claimed."
)
