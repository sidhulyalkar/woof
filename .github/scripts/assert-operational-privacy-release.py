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
    "process.env.WOOF_RELEASE_SHA",
)
require(
    "apps/api/src/observability/observability.service.ts",
    "release: resolveReleaseIdentity()",
    "const release = resolveReleaseIdentity()",
    "release,",
)
require(
    "apps/api/src/sentry.ts",
    "release: resolveReleaseIdentity()",
    "sendDefaultPii: false",
    "scrubSentryEvent",
)

require(
    "apps/web/src/lib/observability/sentry-policy.ts",
    "NEXT_PUBLIC_WOOF_RELEASE_SHA",
    "NEXT_PUBLIC_SENTRY_REPLAY_ENABLED",
    "UNKNOWN_RELEASE",
    "sessionSampleRate: enabled ? 0.01 : 0",
    "errorSampleRate: enabled ? 0.1 : 0",
)
require(
    "apps/web/sentry.client.config.ts",
    "release: resolveWebReleaseIdentity()",
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
    require(path, "release: resolveWebReleaseIdentity()")

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
        '--build-arg WOOF_RELEASE_SHA="${GITHUB_SHA}"',
        "NEXT_PUBLIC_WOOF_RELEASE_SHA: ${{ github.sha }}",
        "NEXT_PUBLIC_SENTRY_REPLAY_ENABLED: 'false'",
        "Enforce deployed API release identity",
    )

print("Operational privacy contract preserves exact release identity and privacy-closed replay.")
