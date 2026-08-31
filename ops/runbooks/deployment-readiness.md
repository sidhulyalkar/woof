# Deployment and readiness incident runbook

Runbook ID: `deployment-readiness`  
Repository qualification: `CODE_QUALIFIED`  
Live rehearsal status: `NOT_YET_PROVEN`

## Authority boundary

This runbook is executable guidance for Woof's maintained production workflow and observability endpoints. It is not evidence that the Fly.io API app, Vercel Web app, production secrets, or a real rollback target are currently available.

The maintained production API deployment target is `woof-api-prod`. The deployment workflow requires readiness to return the exact deployed Git revision before it accepts API rollout success. A green CI build is not a production deployment.

Never paste `OPS_METRICS_TOKEN`, JWT material, provider credentials, database URLs, or other secrets into incident notes or shell history.

## Detection

Use this runbook for these alert-policy keys:

- `telemetryMissing`
- `unknownRelease`
- `readinessFailures`
- `http5xx`
- `application5xxBurst`
- `requestDurationInvalid`

Start with the public readiness boundary:

```bash
export WOOF_API_ORIGIN="https://woof-api-prod.fly.dev"
curl --fail-with-body --silent --show-error \
  "$WOOF_API_ORIGIN/api/v1/ops/health/ready" | jq
```

Expected healthy shape includes `status: "ready"` and a non-placeholder `releaseSha`.

If operational metrics access is available, read it without printing the token:

```bash
test -n "${OPS_METRICS_TOKEN:-}" || { echo "OPS_METRICS_TOKEN unavailable" >&2; exit 2; }
curl --fail-with-body --silent --show-error \
  -H "x-woof-ops-token: ${OPS_METRICS_TOKEN}" \
  "$WOOF_API_ORIGIN/api/v1/ops/metrics.json" | jq
```

`/ops/metrics` and `/ops/metrics.json` intentionally fail closed when the operational token is absent. Do not weaken the guard during an incident.

## Impact

Treat readiness failure as a release-admission problem, not automatically as total process death. Liveness, database readiness, release identity, and user-facing requests answer different questions.

Treat missing telemetry as an observability incident even when user traffic appears healthy. Do not declare recovery from a dashboard that stopped receiving data.

Treat an unknown release identity as a provenance incident. Do not assume the running image corresponds to the latest `main` commit.

## Containment

1. Stop manual production deploys until the running release identity is known.
2. Do not merge a new release solely to overwrite an unexplained bad release.
3. If an active deploy is failing readiness, preserve the deployment/release logs before retrying.
4. If the failure began immediately after a deploy and the previous image is known-good, prepare an image rollback only after the database-compatibility gate below is satisfied.
5. If the failure is isolated to one optional integration, prefer disabling that integration through its qualified configuration boundary over rolling back the entire application.

## Diagnosis

Record these facts before changing state:

```bash
export FLY_APP="woof-api-prod"
fly status --app "$FLY_APP"
fly releases --app "$FLY_APP" --image
```

Also record:

- alert key and first observed time;
- current `releaseSha` from `/ops/health/ready` when reachable;
- intended Git SHA from the deployment workflow or release record;
- whether database migrations ran for the suspect release;
- whether failure affects readiness, user requests, Web only, API only, or an optional provider;
- the smallest reproducible user-facing request that demonstrates the incident, without private user data.

If `/health/live` is reachable while `/health/ready` is not, inspect database/release readiness before assuming process failure:

```bash
curl --fail-with-body --silent --show-error \
  "$WOOF_API_ORIGIN/api/v1/ops/health/live" | jq
```

Do not use HTTP health counters to infer WebSocket/realtime health. Realtime has a separate manual/deferred runbook until its dedicated metric exists.

## Recovery

### Forward recovery

Prefer a forward fix when:

- the suspect release ran a schema migration that may not be backward compatible;
- the previous image is unavailable;
- the failure is configuration-specific and can be corrected without code rollback;
- a small, reviewed fix has lower blast radius than reverting application state.

After deploying a fix, require readiness to identify the exact intended revision. The maintained production workflow already enforces this for automated API deploys.

### Previous-image recovery

Fly rollback is a redeployment of a prior image, not database time travel. List exact release images:

```bash
fly releases --app "$FLY_APP" --image
```

Set `PRIOR_IMAGE` only after identifying a known-good release:

```bash
export PRIOR_IMAGE="registry.fly.io/woof-api-prod:deployment-REPLACE_WITH_VERIFIED_IMAGE"
```

Before redeploying it, perform the database compatibility gate:

- Did the failed release run any migration?
- Can the previous image operate safely against the current schema?
- Would its configured `release_command` try to run an old migration set?

If and only if the current schema is verified compatible with the prior image, redeploy the exact image and skip the release command so rollback does not rerun old migrations:

```bash
fly deploy \
  --app "$FLY_APP" \
  --config apps/api/fly.toml \
  --image "$PRIOR_IMAGE" \
  --remote-only \
  --skip-release-command
```

If schema compatibility is unknown, **do not execute this rollback**. Escalate to `database-migration.md` and use a reviewed forward/compensating migration plan.

## Rollback

There is intentionally no `fly releases rollback` command in this runbook. Woof rollback authority is the exact prior container image plus a database-compatibility decision.

Do not:

- run `prisma migrate reset` in production;
- delete migration-history rows to force a release through;
- deploy an older image and assume the database reverted with it;
- use `--strategy immediate` merely to make an uncertain rollback faster;
- accept a release that is healthy but reports the wrong `releaseSha`.

## Verification

Recovery is not complete until all applicable checks pass:

1. `/api/v1/ops/health/live` succeeds.
2. `/api/v1/ops/health/ready` succeeds.
3. `releaseSha` equals the intended deployed Git revision.
4. Guarded operational metrics are available when they were configured before the incident.
5. The triggering alert condition has cleared for at least one full alert evaluation window.
6. A representative authenticated API journey succeeds without private test fixtures.
7. If Web was involved, its production endpoint resolves and talks to the intended API origin.
8. No new migration or provider failure was introduced during recovery.

## Evidence

Capture only low-sensitivity operational evidence:

- incident start/end timestamps;
- alert names and threshold states;
- intended Git SHA and observed `releaseSha`;
- Fly release version and image identifier;
- deployment workflow run ID;
- migration identifiers, never database contents;
- command exit codes and bounded health responses;
- recovery image/fix SHA;
- verification checklist result.

Do not copy request bodies, access tokens, push subscription material, health/behavior provider bodies, object keys, or user identifiers into the incident artifact.

## Rehearsal

Live rehearsal status: `NOT_YET_PROVEN`

Promotion to `PROVEN` requires a recorded production-like or authorized production drill that demonstrates exact release identification, a safe recovery path, readiness verification, and rollback behavior with the database compatibility decision explicitly exercised.
