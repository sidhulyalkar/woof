# dogOS Operational Privacy + Release Identity v1

## Purpose

This contract makes production telemetry answer two questions without gaining product-data authority:

1. **Which exact code release emitted this signal?**
2. **Can operational telemetry observe failures without becoming a second store of private dog, household, health, message, or location context?**

It does not create a new analytics system and it does not claim live production monitoring until deployment authority under #77 is configured and exercised.

## Canonical release identity

`WOOF_RELEASE_SHA` / `NEXT_PUBLIC_WOOF_RELEASE_SHA` is authoritative only when it is an exact 40-character hexadecimal Git commit SHA. Values such as `main`, `latest`, a branch name, a shortened SHA, malformed input, or a missing value resolve visibly to `unknown`.

The same intended GitHub Actions `GITHUB_SHA` is injected into:

- the Fly API image through Docker build argument `WOOF_RELEASE_SHA`;
- API Sentry release metadata;
- API liveness/readiness responses;
- the Vercel Web build through `NEXT_PUBLIC_WOOF_RELEASE_SHA`;
- Web Sentry release metadata.

Production/staging deployment verifies the public API liveness and readiness documents report the exact intended SHA. A healthy HTTP status with a different release identity is a failed deployment qualification.

Release identity is public operational metadata. It must never contain credentials, tokens, environment secrets, user IDs, pet IDs, or arbitrary client-provided labels.

## Session Replay privacy

Browser Session Replay is **disabled by default**.

The build-time switch `NEXT_PUBLIC_SENTRY_REPLAY_ENABLED=true` is required to enable it. The committed production and staging deployment workflows explicitly set it to `false`.

If a future, separately reviewed deployment enables replay:

- normal-session sample rate is capped at 1%;
- error-session sample rate is capped at 10%;
- all text is masked;
- all media is blocked;
- browser Sentry events strip request, user, extra, and breadcrumb payloads before transport.

Enabling replay in code is not permission to collect sensitive content. Product/legal/privacy review and live provider configuration remain separate requirements.

## API telemetry privacy

The existing API Sentry boundary continues to:

- disable default PII;
- drop request/user/extra/breadcrumb payloads;
- strip span data and replace descriptions with bounded operation names;
- ignore expected client errors where appropriate.

Release identity adds attribution, not payload collection.

## Health semantics

Liveness and readiness expose the resolved release SHA so operators can distinguish:

- expected release;
- stale deployment;
- deployment/configuration that failed to carry release identity (`unknown`).

Readiness remains database-backed. A matching SHA cannot make an unavailable database healthy.

## Qualification

`dogOS Operational Privacy + Release CI` proves:

- exact-SHA parsing and fail-visible `unknown` behavior;
- privacy-closed replay defaults and bounded opt-in rates;
- browser event scrubbing;
- API/Web format, lint, typecheck, and production builds;
- deployment source contracts for Git SHA injection and release verification;
- absence of the prior `maskAllText: false`, `blockAllMedia: false`, and 100% error replay configuration.

Existing Production Runtime, Security Baseline, CodeQL, and root CI remain independent release gates when their owned paths are triggered.

## Live evidence still required

After #77 is resolved, operational readiness under #100 still requires:

- a deployed release whose API reports the intended SHA;
- Sentry events attributed to that same SHA;
- external metric scraping and alert routing;
- non-destructive alert/recovery drills;
- backup/restore rehearsal;
- bounded load qualification.

Code qualification is not being described as live operational proof.
