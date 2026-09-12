#!/usr/bin/env python3
"""Fail closed when release identity, promotion authority, or telemetry privacy drifts."""

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
    ".github/scripts/qualify-live-release.mjs",
    "qualifyLiveRelease",
    "woof-live-release-qualification",
    "api-liveness",
    "api-readiness",
    "web-release-provenance",
    "cors-public-origin",
    "RELEASE_RECEIPT_PATH",
    "--self-test",
)

require(
    "infra/docker/Dockerfile.api",
    "ARG WOOF_RELEASE_SHA=unknown",
    "ENV WOOF_RELEASE_SHA=${WOOF_RELEASE_SHA}",
)

require(
    ".github/workflows/deploy-staging.yml",
    "branches: [main]",
    '--build-arg WOOF_RELEASE_SHA="${GITHUB_SHA}"',
    "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ github.sha }}",
    "NEXT_PUBLIC_SENTRY_REPLAY_ENABLED: 'false'",
    "EXPECTED_WEB_ORIGIN: ${{ vars.WEB_ORIGIN }}",
    "Enforce deployed API release identity",
    "Verify deployed Web release and API integration",
    "Qualify live staging release and write receipt",
    "qualify-live-release.mjs",
    "uses: actions/upload-artifact@v7",
    "name: staging-release-${{ github.sha }}",
)

require(
    ".github/workflows/deploy-production.yml",
    "workflow_dispatch:",
    "release_sha:",
    "refs/heads/main",
    'if [[ ! "${RELEASE_SHA}" =~ ^[0-9a-f]{40}$ ]]',
    'git merge-base --is-ancestor "${RELEASE_SHA}" refs/remotes/origin/main',
    "ref: ${{ inputs.release_sha }}",
    '--build-arg WOOF_RELEASE_SHA="${RELEASE_SHA}"',
    "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ env.RELEASE_SHA }}",
    "NEXT_PUBLIC_SENTRY_REPLAY_ENABLED: 'false'",
    "EXPECTED_WEB_ORIGIN: ${{ vars.WEB_ORIGIN }}",
    "Enforce deployed API release identity",
    "Verify deployed Web release and API integration",
    "Qualify live production release and write receipt",
    "qualify-live-release.mjs",
    "uses: actions/upload-artifact@v7",
    "name: production-release-${{ env.RELEASE_SHA }}",
)
reject(
    ".github/workflows/deploy-production.yml",
    "branches: [main]",
    '--build-arg WOOF_RELEASE_SHA="${GITHUB_SHA}"',
    "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ github.sha }}",
)

print(
    "Operational privacy contract preserves exact release identity, explicit production promotion, live receipts, Web/API provenance, and privacy-closed replay."
)
