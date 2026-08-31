# Database and migration incident runbook

Runbook ID: `database-migration`  
Repository qualification: `CODE_QUALIFIED`  
Live rehearsal status: `NOT_YET_PROVEN`

## Authority boundary

Woof production deploys run `pnpm --filter @woof/database db:migrate:deploy` as the Fly release command before application rollout. A successful application image rollback does not reverse schema changes.

There is currently no dedicated privacy-safe database connection-pool saturation gauge. `database_pool_saturation` remains a manual/deferred signal. Readiness must not be described as pool telemetry.

This runbook does not authorize destructive production SQL, migration-history editing, or unreviewed schema rollback.

## Detection

Use this runbook for:

- `readinessFailures`
- `http5xx`
- `application5xxBurst`
- manual/deferred `database_pool_saturation`
- release-command migration failures
- schema incompatibility discovered during deployment rollback analysis

Check liveness and readiness separately:

```bash
export WOOF_API_ORIGIN="https://woof-api-prod.fly.dev"
curl --fail-with-body --silent --show-error \
  "$WOOF_API_ORIGIN/api/v1/ops/health/live" | jq
curl --fail-with-body --silent --show-error \
  "$WOOF_API_ORIGIN/api/v1/ops/health/ready" | jq
```

A readiness failure can indicate database unavailability, but it is not sufficient evidence of connection-pool saturation.

## Impact

Classify the incident before recovery:

- **Migration admission failure:** release command failed before new Machines became authoritative.
- **Runtime database outage:** previously healthy release cannot reach or use the database.
- **Schema/application mismatch:** application revision and schema are individually reachable but incompatible.
- **Suspected saturation:** latency/timeouts suggest pressure, but no dedicated pool metric exists yet.
- **Data-integrity concern:** writes may have partially succeeded or invariants may be violated. Escalate immediately and stop speculative repair.

## Containment

1. Stop subsequent application deploys while migration state is unknown.
2. Do not run another migration merely because the first attempt failed.
3. Preserve release-command logs and the migration name that was executing.
4. If writes may be unsafe, disable the narrow affected product surface when a qualified feature/config boundary exists. Do not globally disable unrelated reads without evidence.
5. Do not use a previous application image until schema compatibility with that image has been reviewed.
6. Never expose `DATABASE_URL` in incident notes, terminal transcripts, screenshots, or tickets.

## Diagnosis

Record:

- current application `releaseSha` when readiness is reachable;
- intended release SHA;
- exact migration directory names introduced by the suspect release;
- whether the release command completed, failed, or timed out;
- whether failure affects reads, writes, both, or one bounded operation;
- whether the current schema is backward compatible with the previous image.

Inspect the release history without changing state:

```bash
export FLY_APP="woof-api-prod"
fly releases --app "$FLY_APP" --image
```

Inspect repository migration history from the exact release source. Do not infer production migration state solely from the repository.

For suspected pool saturation, collect bounded evidence such as request latency, database timeout classes, and provider/platform connection telemetry if production authority exposes it. Because Woof has no maintained pool gauge yet, mark the conclusion as `UNCONFIRMED_POOL_PRESSURE` unless direct database/platform evidence exists.

## Recovery

### Migration failed before rollout

Prefer fixing the migration or environment and re-running the normal deployment path. Prisma `migrate deploy` should remain the authority for applying committed production migrations.

A failed migration must be understood before retry. If the migration is non-transactional or can partially apply, determine actual database state first through authorized database tooling.

### Application/schema mismatch after rollout

Prefer a forward-compatible application fix or a reviewed compensating migration. The recovery should make both the current and immediately previous application version compatible whenever practical, restoring rollback room.

### Database unavailable

Restore database/network/provider availability first. Do not mask a database outage by weakening readiness.

### Suspected saturation

Reduce load only through bounded, reversible mechanisms. Examples include disabling a high-volume optional worker or import path through its maintained feature/config boundary. Do not introduce an unreviewed global rate limit during the incident unless there is no safer containment path.

## Rollback

Database rollback is **not automatic** when redeploying a prior Fly image.

Before application image rollback, answer all three:

1. What schema changes have already reached the database?
2. Is the prior image compatible with that schema?
3. Can the prior image be deployed with the release command skipped so its older migration set does not run?

When the answer is verified yes, follow `deployment-readiness.md` and use the exact prior image with `--skip-release-command`.

If schema compensation is required, ship it as a reviewed migration with explicit forward behavior. Do not:

- run `prisma migrate reset`;
- drop the production database or schema;
- delete or edit Prisma migration-history rows as an incident shortcut;
- manually mark a failed migration successful without verifying actual schema state;
- reverse a destructive migration from memory;
- restore a backup over live production without a separate authorized recovery plan and impact review.

## Verification

Require:

1. release command/migration step exits successfully for the intended release;
2. `/ops/health/ready` is healthy and reports the intended `releaseSha`;
3. representative reads succeed;
4. representative writes succeed only in non-private test fixtures or controlled synthetic records;
5. affected alert windows clear;
6. no unexpected migration remains pending for the release;
7. application version immediately before the incident is classified as compatible or incompatible with the recovered schema;
8. any manual saturation conclusion is labeled with the evidence source rather than promoted from readiness alone.

## Evidence

Capture:

- release SHA and Fly release/image identifier;
- migration directory names;
- release-command status and bounded error class;
- schema-compatibility decision for rollback;
- recovery migration/fix SHA;
- health/readiness results;
- alert clearance time.

Never capture row contents, private object keys, credentials, connection strings, or raw database/provider exception payloads unless a separately authorized forensic process requires them.

## Rehearsal

Live rehearsal status: `NOT_YET_PROVEN`

Promotion to `PROVEN` requires a production-like drill that demonstrates a failed release command or schema mismatch, prevents destructive shortcuts, exercises the compatibility gate, and restores a revision that passes readiness and a synthetic read/write journey.
