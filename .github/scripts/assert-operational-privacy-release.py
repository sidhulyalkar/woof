#!/usr/bin/env python3
"""Fail closed when release identity or telemetry privacy authority drifts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    target = ROOT / path
    if not target.is_file():
        raise SystemExit(f"required operational privacy source missing: {path}")
    return target.read_text()


def require(path: str, *markers: str) -> None:
    text = read(path)
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise SystemExit(f"{path}: missing required markers: {missing}")


def reject(path: str, *markers: str) -> None:
    text = read(path)
    present = [marker for marker in markers if marker in text]
    if present:
        raise SystemExit(f"{path}: forbidden operational privacy markers: {present}")


require(
    "apps/api/src/observability/release-identity.ts",
    "GIT_SHA_PATTERN",
    "UNKNOWN_RELEASE",
    "resolveReleaseIdentity(value: string | undefined)",
    "resolveProcessReleaseIdentity()",
    "process.env.WOOF_RELEASE_SHA",
)
require(
    "apps/api/src/observability/observability.service.ts",
    "release: resolveProcessReleaseIdentity()",
    "const release = resolveProcessReleaseIdentity()",
    "release,",
)
require(
    "apps/api/src/sentry.ts",
    "release: resolveProcessReleaseIdentity()",
    "sendDefaultPii: false",
    "scrubSentryEvent",
)

require(
    "apps/web/src/lib/observability/sentry-policy.ts",
    "resolveWebReleaseIdentity(value: string | undefined)",
    "resolveWebRuntimeReleaseIdentity()",
    "NEXT_PUBLIC_WOOF_RELEASE_SHA",
    "NEXT_PUBLIC_SENTRY_REPLAY_ENABLED",
    "UNKNOWN_RELEASE",
    "sessionSampleRate: enabled ? 0.01 : 0",
    "errorSampleRate: enabled ? 0.1 : 0",
)
require(
    "apps/web/src/app/layout.tsx",
    "resolveWebRuntimeReleaseIdentity()",
    "'woof-release'",
    "'woof-api-origin'",
    "NEXT_PUBLIC_API_URL",
)
require(
    "apps/web/sentry.client.config.ts",
    "release: resolveWebRuntimeReleaseIdentity()",
    "maskAllText: true",
    "blockAllMedia: true",
    "scrubBrowserSentryEvent",
)
reject(
    "apps/web/sentry.client.config.ts",
    "maskAllText: false",
    "blockAllMedia: false",
    "replaysOnErrorSampleRate: 1",
)
for path in ["apps/web/sentry.server.config.ts", "apps/web/sentry.edge.config.ts"]:
    require(path, "release: resolveWebRuntimeReleaseIdentity()")

require(
    ".github/scripts/verify-web-deployment-provenance.mjs",
    "verifyWebDeploymentProvenance",
    "woof-release",
    "woof-api-origin",
    "EXPECTED_RELEASE",
    "EXPECTED_API_URL",
    "--self-test",
)
require(
    ".github/scripts/verify-live-release.mjs",
    "assertReleaseBody",
    "x-content-type-options",
    "access-control-allow-origin",
    "auth/me",
    "EXPECTED_RELEASE",
    "--self-test",
)
require(
    ".github/scripts/write-release-receipt.mjs",
    "schemaVersion: 2",
    "SHA_PATTERN",
    "buildReleaseReceipt",
    "production receipt requires successful exact-SHA staging qualification",
    "canonicalReleaseChecksVerified",
    "apiReleaseIdentityVerified: true",
    "webReleaseIdentityVerified: true",
    "webApiOriginVerified: true",
    "liveBlackBoxVerified",
    "webDeploymentUrl",
    "parsed.username = '';",
    "parsed.password = '';",
    "--self-test",
)
reject(
    ".github/scripts/write-release-receipt.mjs",
    "FLY_API_TOKEN",
    "VERCEL_TOKEN",
    "DATABASE_URL",
    "JWT_SECRET",
    "Authorization",
)

require(
    "infra/docker/Dockerfile.api",
    "ARG WOOF_RELEASE_SHA=unknown",
    "ENV WOOF_RELEASE_SHA=${WOOF_RELEASE_SHA}",
)
for path in [
    ".github/workflows/deploy-production.yml",
    ".github/workflows/deploy-staging.yml",
]:
    require(
        path,
        '--build-arg WOOF_RELEASE_SHA="${RELEASE_SHA}"',
        "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ env.RELEASE_SHA }}",
        "NEXT_PUBLIC_SENTRY_REPLAY_ENABLED: 'false'",
        "STABLE_WEB_ORIGIN: ${{ vars.WEB_ORIGIN }}",
        "Enforce deployed API release identity",
        "Verify immutable Web deployment release and API integration",
        "Verify stable",
        "Web origin release and API integration",
        "Run non-destructive live release smoke",
        "verify-web-deployment-provenance.mjs",
        "verify-live-release.mjs",
        "CANONICAL_RELEASE_CHECKS_VERIFIED: 'true'",
        "LIVE_BLACK_BOX_VERIFIED: 'true'",
        "write-release-receipt.mjs",
        "actions/upload-artifact@v7",
    )

print(
    "Operational privacy contract preserves exact API/Web release identity, stable-origin provenance, "
    "non-destructive live verification, and privacy-safe release receipts."
)
