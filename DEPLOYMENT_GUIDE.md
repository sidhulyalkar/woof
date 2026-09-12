# Woof Production Deployment Guide

## Status

**Pre-public-beta deployment authority.** This guide describes the current release path. It is not evidence that Woof is already live.

Woof keeps four claims separate:

> repository-qualified ≠ deployed ≠ device-qualified ≠ pilot-validated

A successful pull request proves repository behavior. A successful staging workflow proves one exact commit was deployed and live-smoke-qualified in staging. A successful production workflow proves that same staging-qualified commit was promoted to the configured production providers. Physical iPhone/TestFlight evidence and real-user pilot evidence remain separate gates.

## Canonical architecture

```text
qualified PR
    |
    v
protected main
    |
    v
exact-SHA canonical main checks
    |
    v
Deploy to Staging
    |-- managed staging PostgreSQL
    |-- Fly app: woof-api-staging
    |-- separate Vercel staging project
    |-- stable staging WEB_ORIGIN
    |-- migration + live/ready + Web provenance + live smoke
    `-- 30-day staging receipt
    |
    v
backup / restore + staging black-box qualification
    |
    v
manual Deploy to Production(release_sha)
    |-- re-verifies exact-SHA main checks
    |-- requires successful staging run for that SHA
    |-- managed production PostgreSQL
    |-- Fly app: woof-api-prod
    |-- separate Vercel production project
    |-- stable production WEB_ORIGIN
    |-- migration + live/ready + Web provenance + live smoke
    `-- 90-day production receipt
```

Do not deploy from a laptop or from an uncommitted checkout. Do not manually mutate release identity. Production is a promotion of an exact Git SHA.

## 1. Repository authority

Before public beta, apply the `main` ruleset tracked in issue #80:

- pull requests required for ordinary changes;
- force pushes and branch deletion disabled;
- required conversations resolved;
- branches current before merge or equivalent merge-queue semantics;
- minimal explicit emergency bypass;
- required unconditional checks: root CI, Security Baseline, CodeQL, and Release Promotion Authority.

The connected ChatGPT GitHub App does not have repository-administration authority, so this is an owner/admin action.

## 2. Provider isolation

Create separate resources for staging and production. Sharing credentials, databases, or the same Vercel project makes the staging boundary cosmetic rather than protective.

### PostgreSQL

Provision two managed PostgreSQL databases or isolated projects:

- staging database for `woof-api-staging`;
- production database for `woof-api-prod`.

The provider must support an operator-visible backup/snapshot mechanism suitable for the backup and restore rehearsal in issue #100. Keep production and staging credentials distinct. `DATABASE_URL` is application runtime authority and belongs in Fly secrets, never GitHub source or Vercel.

Do **not** run development migration commands or manually seed production. Fly uses the committed release command:

```text
pnpm --filter @woof/database db:migrate:deploy
```

### Fly.io

Create or verify these applications:

```text
woof-api-staging
woof-api-prod
```

The committed `apps/api/fly.toml` is authoritative for port 4000, HTTPS, rolling deploys, the migration release command, one warm Machine minimum, and `/api/v1/ops/health/ready` traffic readiness.

Use separately scoped deploy tokens for staging and production. GitHub receives deploy authority through environment-scoped `FLY_API_TOKEN`; the Fly applications themselves receive runtime secrets directly through Fly secret management.

### Vercel

Create **separate Vercel projects** for the Web client, for example:

```text
woof-web-staging
woof-web-production
```

Both projects use `apps/web` from this repository. The staging workflow intentionally builds with production semantics and deploys with `--prod` **inside the isolated staging project**. This produces a stable staging project alias while keeping it independent from the real production project.

Never point the GitHub `staging` environment at the production Vercel project merely to make a workflow green.

## 3. Stable Web origins and CORS

Each GitHub environment requires a non-secret environment variable named `WEB_ORIGIN` containing the exact stable HTTPS browser origin, with no path, query, or fragment.

Examples only:

```text
staging   WEB_ORIGIN=https://woof-web-staging.vercel.app
production WEB_ORIGIN=https://app.example.com
```

Use the real aliases/domains assigned to the projects rather than copying these examples blindly.

The matching Fly app must receive `CORS_ORIGIN` with the same stable Web origin. Multiple explicit HTTPS origins may be comma-separated when genuinely required. Wildcard CORS is forbidden in production.

The release workflow live-smoke-checks that the configured stable origin receives the expected credentialed CORS response.

## 4. API runtime configuration

At minimum, each Fly application needs production-shaped runtime configuration for:

```text
DATABASE_URL
JWT_SECRET
CORS_ORIGIN
```

`JWT_SECRET` must be a non-development secret of at least 32 characters. `NODE_ENV=production`, `PORT=4000`, `API_PREFIX=api/v1`, and `API_DOCS_ENABLED=false` are committed in `apps/api/fly.toml`.

Configure optional systems only when their complete authority exists:

- `SENTRY_DSN` for live error monitoring;
- `OPS_METRICS_TOKEN` for protected metric scraping;
- `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_PUBLIC_URL` for private media storage;
- `VAPID_PUBLIC_KEY` + `VAPID_PRIVATE_KEY` + valid `CONNECTOR_CREDENTIALS_KEY` for Web Push;
- `OPENAI_API_KEY` for enabled model-backed Health Lens behavior;
- complete Behavior Vision URL/token/release pin if that service is enabled;
- connector encryption authority before enabling dogOS connectors.

Unavailable optional providers must remain explicit unavailable/degraded states. Do not add placeholder credentials solely to pass startup.

## 5. GitHub environment authority

Create GitHub environments named exactly:

```text
staging
production
```

Configure **staging** with:

```text
secret FLY_API_TOKEN
secret VERCEL_TOKEN
secret VERCEL_ORG_ID
secret VERCEL_PROJECT_ID
variable WEB_ORIGIN
```

Configure **production** with the same five values, but bound to production resources. `SLACK_WEBHOOK` is optional.

The current Vercel account must first contain the intended Woof projects. Do not reuse an unrelated project ID. Production should require explicit environment reviewer approval before jobs can consume production authority.

## 6. Canonical exact-SHA release checks

Before staging or production can touch provider credentials, `.github/scripts/verify-main-release-checks.py` requires successful `push` runs for the exact candidate SHA from:

```text
ci.yml
security-baseline-ci.yml
codeql.yml
release-promotion-authority-ci.yml
```

These workflows are intentionally unconditional on `main`. Path-scoped feature lanes remain valuable domain evidence but are not global release blockers because an unrelated PR must not deadlock on a workflow that never ran.

## 7. Staging release

A merge to `main` automatically starts `Deploy to Staging`. The workflow:

1. resolves an exact 40-character SHA;
2. proves the checkout matches and is reachable from `origin/main`;
3. waits for the canonical exact-SHA main checks;
4. validates staging deployment authority;
5. deploys `woof-api-staging` with the exact release SHA;
6. runs the committed Prisma migration release command;
7. requires `/ops/health/live` and database-backed `/ops/health/ready` to report that SHA;
8. builds and deploys production-shaped Web artifacts to the isolated staging Vercel project;
9. verifies both the immutable Vercel URL and stable staging origin expose the expected Web release/API provenance;
10. runs the non-destructive live release smoke;
11. retains a privacy-safe staging receipt for 30 days.

A failed staging deploy is not permission to bypass a gate. Diagnose the first failing boundary and repair that boundary.

## 8. Automated live smoke semantics

`.github/scripts/verify-live-release.mjs` verifies without creating user data:

- API liveness is HTTP 200 and reports the selected SHA;
- database-backed readiness is HTTP 200 and reports the selected SHA;
- production security headers include `X-Content-Type-Options: nosniff`;
- the real stable Web origin receives the configured credentialed CORS authority;
- unauthenticated `/auth/me` fails closed with HTTP 401;
- Swagger/API documentation is not publicly reachable in the production-shaped runtime.

Repository tests own negative/hostile CORS behavior. The deploy smoke intentionally avoids manufacturing a production server error merely to prove a known-invalid origin is rejected.

## 9. Staging black-box acceptance

Before the first public-beta production promotion, exercise the following against staging with synthetic accounts and retain a bounded pass/fail record, not private payloads:

```text
register -> login -> persisted session -> logout -> revoked session denied
Guardian -> create pet -> Today -> Daily Signals -> Adventure -> Story/Field Journal
caregiver invite -> accept -> authorized read/write -> revoke -> immediate denial
Community/Packs -> approved locality -> visibility -> reactions -> cohort privacy
realtime -> connect -> authorized conversation -> block/revoke -> authority removed
Health Lens -> normal path -> model unavailable -> conservative fallback
connector unavailable -> explicit unavailable state, never fabricated connectivity
Media Library -> upload/read/delete where storage is configured
account deletion -> database + owned external-object deletion
```

Also test expired session, duplicate mutation/retry, interrupted network, DB/provider degradation where safely reproducible, and production Web CORS from the actual stable origin.

## 10. Backup and restore rehearsal

Production promotion for public beta is blocked until issue #100 Phase 4 has real evidence.

For the selected database provider:

1. identify the actual backup/snapshot authority and retention policy;
2. define RPO/RTO as operator goals, not guarantees;
3. restore a real or production-equivalent snapshot into an isolated non-production database;
4. apply the full committed migration chain/checks;
5. verify auth/session records, pet ownership, caregiver grants, Daily Signals/Story schemas, and deletion/privacy invariants;
6. record date, snapshot identity, target environment, release SHA, duration, and pass/fail without row contents or identifiers.

“Backups enabled” is not restore evidence.

## 11. Production promotion

Production is manual-only. Select the exact SHA whose staging receipt and staging black-box evidence you intend to promote, then dispatch `Deploy to Production` with that full SHA.

Before credentials are consumed, production independently re-verifies:

- exact SHA syntax and checkout identity;
- `main` ancestry;
- canonical exact-SHA main release checks;
- at least one successful `Deploy to Staging` run for exactly that SHA.

The production jobs then repeat API migration/release identity, stable Web provenance, CORS, fail-closed auth, docs-disabled, and live-smoke checks before a 90-day production receipt can exist.

Do not reinterpret a partial Fly-only or Vercel-only deployment as a complete Woof release.

## 12. Live operational qualification

Immediately after the first successful production release, close issue #100 Phase 6 with real provider evidence:

- send one deliberate benign test event through production error monitoring and verify the exact release SHA;
- externally scrape/aggregate protected operational metrics;
- perform one non-destructive alert drill that reaches the accountable operator;
- continuously observe `/ops/health/live` and `/ops/health/ready`;
- verify the operator can reach the committed incident runbooks;
- prove monitoring/provider failure becomes an explicit degraded-observability state rather than silent green.

Only then describe live observability as proven.

## 13. Rollback

Application rollback is an explicit new production promotion of a previously staging-qualified exact `main` SHA. Never redeploy an unknown local checkout and never rewrite release identity.

Database rollback is separate. Migrations are forward-safe authority; do not delete Prisma migration history, manually mark failed migrations successful, or restore over live production without a reviewed recovery plan.

Before each production promotion, identify the prior known-good staging-qualified application SHA as the application rollback target and confirm the new migration remains compatible with it when rollback may be needed.

## 14. Native distribution is a separate gate

Web/API production success does not make the iOS client device-qualified. Before public beta, separately prove:

- signed Release build on a physical iPhone;
- install/launch and upgrade behavior;
- TestFlight or equivalent distribution;
- auth persistence and logout/account deletion;
- offline/poor-network recovery;
- notification/photo/camera permission states where used;
- safe areas and keyboard overlap;
- Dynamic Type/large text;
- VoiceOver on critical paths.

The existing Xcode simulator qualification is repository evidence, not a substitute for this gate.

## 15. Pilot and launch boundary

After deployment, recovery, observability, and device gates, run a small owner/advisor pilot. Measure whether Woof reduces relationship-management burden and improves useful actions rather than optimizing raw engagement.

Public beta is justified only when one explicitly identified release is:

```text
protected-source qualified
+ staging deployed and black-box qualified
+ backup/restore rehearsed
+ manually promoted to production
+ production observable and recoverable
+ physical-device distributed
+ privacy boundaries verified
+ small-pilot validated
```

Until all of those have evidence, use the narrower truthful status that applies.

## Explicitly retired instructions

The old 2025 guide contained instructions that are no longer authoritative. Do not:

- run interactive/development Prisma migration commands against production;
- manually seed production merely to create test users;
- expose Swagger in production;
- hand-deploy from a laptop as the canonical release path;
- reuse an unrelated Vercel project;
- rely on old `/api/v1/health` paths;
- claim PWA installation/offline support unless it is deliberately rebuilt and requalified;
- claim the app is production-ready because provider accounts exist.

The GitHub workflows, committed runtime config, release receipts, runbooks, and this guide now define deployment authority.
