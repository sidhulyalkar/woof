# Woof production deployment guide

This runbook describes a safe production rollout for the Woof monorepo. It intentionally separates local-development conveniences from production requirements.

> `docker-compose.yml` is a **local development stack only**. Its published database/Redis ports and default local credentials must never be used as a production deployment configuration.

## 1. Release prerequisites

Before deploying a release candidate:

- The exact candidate SHA must pass the repository's required CI.
- For Adventure System releases, the exact SHA must pass `.github/workflows/adventure-system-ci.yml`.
- PostgreSQL must support the `vector` extension used by the migration chain.
- A recoverable database backup/snapshot must exist before migrations are applied.
- Production secrets must be stored in the hosting provider's encrypted secret store, not committed files.
- `CORS_ORIGIN` must contain only intended production web origins.
- `ENABLE_ADVENTURE_SYSTEM` should remain `false` for the first application deployment of an Adventure migration.

Do **not** seed a production database automatically as part of deployment.

## 2. Required production configuration

At minimum, the API needs:

```env
NODE_ENV=production
PORT=4000
API_PREFIX=api/v1
DATABASE_URL=postgresql://...
JWT_SECRET=<strong random secret, at least 32 characters>
CORS_ORIGIN=https://app.example.com
ENABLE_ADVENTURE_SYSTEM=false
API_DOCS_ENABLED=false
```

The API validates configuration at startup and rejects known development-style JWT secrets in production.

Configure optional integrations only when they are actually used:

```env
SENTRY_DSN=...

# Private media storage
S3_ENDPOINT=...
S3_BUCKET=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
AWS_REGION=...

# Web push
VAPID_PUBLIC_KEY=...
VAPID_PRIVATE_KEY=...

# Health Lens model
OPENAI_API_KEY=...
OPENAI_HEALTH_MODEL=...

# Optional Behavior Vision service
BEHAVIOR_VISION_SERVICE_URL=...
BEHAVIOR_VISION_SERVICE_TOKEN=...
```

When `BEHAVIOR_VISION_SERVICE_URL` is configured in production, its token is required. Partial storage credentials are rejected. Media derivatives require complete private object-storage configuration.

### API documentation

Swagger is available automatically in development. In production it is disabled by default. Only set:

```env
API_DOCS_ENABLED=true
```

when exposing production API documentation is an explicit operational decision.

## 3. Database migration procedure

Use the non-interactive production migration command:

```bash
pnpm install --frozen-lockfile
pnpm --filter @woof/database db:generate
pnpm --filter @woof/database db:migrate:deploy
```

Never run an interactive development migration command against production.

After migration, verify schema parity from a controlled release environment:

```bash
pnpm --filter @woof/database exec prisma migrate diff \
  --from-url "$DATABASE_URL" \
  --to-schema-datamodel prisma/schema.prisma \
  --exit-code
```

For the Adventure release, migrations are intentionally additive. Do not automatically generate or run a destructive rollback that drops reward-history tables.

## 4. Build the release candidate

From the repository root:

```bash
pnpm install --frozen-lockfile
pnpm --filter @woof/database db:generate
pnpm --filter @woof/api type-check
pnpm --filter @woof/web type-check
pnpm --filter @woof/api build
NEXT_PUBLIC_API_URL=https://api.example.com/api/v1 pnpm --filter @woof/web build
```

The API and web production builds must succeed without `ignoreBuildErrors`, disabled type checking, or equivalent escape hatches.

## 5. Deploy the API with features dark

Deploy the API application with:

```env
ENABLE_ADVENTURE_SYSTEM=false
```

The Adventure and Pack public routes should remain unavailable during the first post-migration verification pass.

The process exposes two operational health semantics:

- `GET /api/v1/health/live` checks whether the API process is alive and does not depend on PostgreSQL.
- `GET /api/v1/health/ready` performs a real lightweight PostgreSQL probe and returns HTTP 503 if the required database dependency is unavailable.
- `GET /api/v1/health` remains a compatibility alias for readiness.

Use **readiness**, not liveness, when deciding whether a replica should receive user traffic.

Example:

```bash
curl --fail https://api.example.com/api/v1/health/live
curl --fail https://api.example.com/api/v1/health/ready
```

Do not route production traffic to a replica that is live but not ready.

## 6. Deploy the web application

The web build requires the public API base URL at build/runtime as appropriate for the hosting environment:

```env
NEXT_PUBLIC_API_URL=https://api.example.com/api/v1
```

For Vercel, configure the monorepo project so installation happens from the repository root and the `@woof/web` package is built with the workspace lockfile. Do not replace the frozen workspace install with an independent `npm install` in `apps/web`.

## 7. Post-deploy smoke tests

Before Adventure is enabled, verify:

1. `/health/live` returns 200.
2. `/health/ready` returns 200 and reports the database check as `up`.
3. Registration/login and authenticated session refresh work.
4. A user cannot read or mutate another user's pet.
5. Private Media Library assets remain owner-scoped and signed URLs are ephemeral.
6. Activity logging persists even if optional reward/telemetry integration is unavailable.
7. Health Lens emergency results suppress game navigation and do not produce XP treatment.
8. Adventure routes are unavailable while `ENABLE_ADVENTURE_SYSTEM=false`.
9. Sentry/application logs contain no secrets, OAuth access tokens, or private signed media URLs.

## 8. Enable Adventure deliberately

Only after the dark deployment is healthy:

1. Set `ENABLE_ADVENTURE_SYSTEM=true` on a controlled production rollout.
2. Verify `GET /api/v1/adventure/me` for a test account.
3. Select and complete a low-risk quest.
4. Verify one CareEvent and one RewardLedger receipt are produced for the dedupe key.
5. Retry the same completion and verify no second reward is minted.
6. Verify the Pawprint Compass describes opportunity coverage rather than health status.
7. Verify `SAFE_OPT_OUT` produces Bond progress where eligible.
8. Verify emergency/illness flows remain outside game rewards.
9. Observe structured reward-decision and application-error telemetry during rollout.

## 9. Rollback strategy

The Adventure database changes are additive and the RewardLedger is historical evidence. During an incident:

1. Set `ENABLE_ADVENTURE_SYSTEM=false` to remove Adventure/Pack public access.
2. Roll application code back to the last qualified release if necessary.
3. Leave additive Adventure tables and reward history intact by default.
4. Investigate and repair forward unless an explicitly approved data-removal procedure is required.

Do not drop Adventure tables merely to roll back application behavior.

## 10. Production database and infrastructure guidance

Use a managed PostgreSQL service with:

- encrypted connections,
- automated backups and point-in-time recovery where available,
- restricted network access,
- separate least-privilege application credentials,
- pgvector support,
- monitoring for connections, storage, latency, and failed queries.

Redis, n8n, or other auxiliary services must likewise use production credentials and private networking. The root `docker-compose.yml` is not a production blueprint.

## 11. Observability

At minimum, production should capture:

- API exceptions and release SHA,
- readiness failures,
- reward decisions without direct user/pet identifiers,
- reward-emission failures after activity persistence,
- database/migration failures,
- Health Lens model/provider failures with sensitive inputs redacted,
- Behavior Vision provider failures with fail-closed behavior,
- web runtime exceptions.

Never log raw authentication tokens, OAuth provider tokens, private signed media URLs, or secrets.

## 12. Final release evidence

For Adventure System releases, record in PR #6 before changing draft status:

- final candidate SHA,
- successful Adventure System CI run,
- migration/parity result,
- full API test result,
- web contract test result,
- API and web build result,
- confirmation that the parent Media/Coach/Health stack has landed and the Adventure branch was requalified on the resulting `main`.

A green stacked-branch run is strong development evidence but is not sufficient final merge evidence until the stack is resolved and requalified.
