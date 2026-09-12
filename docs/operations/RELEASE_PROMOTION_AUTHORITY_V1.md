# Release Promotion Authority v2

> The filename remains stable for existing repository references. This document now describes the v2 production-closure contract.

## Purpose

Woof treats repository qualification, staging qualification, production promotion, device distribution, and pilot validation as different claims.

A green pull request does not mean a release is deployed. A merge to `main` does not mean a release is production. Production is an explicit promotion of one exact commit that has passed canonical exact-SHA `main` checks, completed staging successfully, and survived non-destructive live release verification.

## Authority model

```text
feature branch
    |
    v
qualified pull request
    |
    v
main
    |
    +--> exact-SHA canonical main checks
    |      - root CI
    |      - Security Baseline
    |      - CodeQL
    |      - Release Promotion Authority
    |
    +--> Deploy to Staging waits for those checks
    |      - exact SHA + main ancestry
    |      - isolated staging API/database/Web project
    |      - API migration/deploy
    |      - live + ready exact-SHA verification
    |      - immutable + stable Web provenance
    |      - real stable-origin CORS
    |      - fail-closed auth + docs-disabled live smoke
    |      - privacy-safe staging release receipt
    |
    v
explicit workflow_dispatch(release_sha)
    |
    v
Deploy to Production
       - exact SHA + main ancestry
       - canonical exact-SHA checks re-verified
       - successful exact-SHA staging workflow required
       - production environment authority
       - isolated production API/database/Web project
       - API migration/deploy
       - live + ready exact-SHA verification
       - immutable + stable Web provenance
       - real stable-origin CORS
       - fail-closed auth + docs-disabled live smoke
       - privacy-safe production release receipt
```

## Canonical main checks

`.github/scripts/verify-main-release-checks.py` owns the minimal unconditional release set:

```text
ci.yml
security-baseline-ci.yml
codeql.yml
release-promotion-authority-ci.yml
```

For the selected exact SHA, each workflow must have a successful `push` run. Pull-request success for another SHA is not accepted as release evidence.

The set is intentionally small and unconditional. Domain workflows that are path-scoped cannot be globally required without risking a deadlock on a commit where they correctly did not run. They remain important PR/domain evidence and may be required by repository rules where GitHub semantics make that safe.

`Release Promotion Authority CI` itself now runs on every pull request to `main` and every push to `main`. It is no longer path-scoped, so staging and production can depend on its presence for every candidate.

## Staging

`Deploy to Staging` runs automatically for pushes to `main`. A manual dispatch re-runs the workflow for the Git ref selected when dispatching; the workflow still rejects that candidate unless its exact SHA is reachable from `origin/main`.

Before any provider credential is consumed, staging:

1. resolves the exact lowercase 40-character candidate SHA;
2. checks out exactly that SHA;
3. proves it is reachable from `origin/main`;
4. waits for every canonical exact-SHA main check above to succeed.

Only then can staging provider jobs begin.

The staging Web workflow is production-shaped but provider-isolated. The GitHub `staging` environment must point to a **dedicated staging Vercel project**. The workflow uses production-mode Vercel build/deploy semantics inside that staging project so a stable staging alias exists for browser CORS and black-box qualification without touching the real production project.

## Production

`Deploy to Production` is manual-only. It does not run on a push to `main`.

The operator provides the exact 40-character release SHA. Before production credentials are used, the workflow proves:

1. the requested SHA resolves exactly;
2. it is reachable from `origin/main`;
3. all canonical exact-SHA main checks are successful;
4. GitHub Actions has a successful `Deploy to Staging` workflow run for that exact SHA.

This makes production independently fail closed even if repository branch protection is temporarily misconfigured. Branch protection remains required as source-governance authority; deployment checks are defense in depth, not a replacement.

## Provider isolation

Staging and production must use separate application/data authority:

```text
staging:
  PostgreSQL staging database/project
  Fly app woof-api-staging
  Vercel staging project
  GitHub staging environment

production:
  PostgreSQL production database/project
  Fly app woof-api-prod
  Vercel production project
  GitHub production environment
```

Do not bind the staging environment to the production Vercel project. Because staging uses `--prod` inside its isolated project to maintain a stable staging alias, a cross-bound project ID would collapse the isolation boundary.

## Stable Web origins

Each GitHub environment provides a non-secret `WEB_ORIGIN` variable containing the exact stable HTTPS browser origin. It must contain no path, query, or fragment.

The corresponding Fly application configures `CORS_ORIGIN` with that stable origin. The live release smoke proves the actual deployed API returns the expected credentialed CORS authority for that origin.

The workflow separately retains the immutable Vercel deployment URL and the stable browser origin. The stable origin is the user-facing browser/CORS authority; the immutable URL is useful deployment provenance.

## Release identity

The selected candidate SHA is the release authority for both API and Web artifacts.

The API image receives:

```text
WOOF_RELEASE_SHA=<exact selected SHA>
```

The Web build receives:

```text
NEXT_PUBLIC_WOOF_RELEASE_SHA=<exact selected SHA>
```

After deployment:

- API `/ops/health/live` and `/ops/health/ready` must report the exact SHA;
- the immutable Web deployment must expose the expected `woof-release` and `woof-api-origin` provenance;
- the stable Web origin must expose the same provenance.

## Live release smoke

`.github/scripts/verify-live-release.mjs` is deliberately non-destructive. It proves:

- process liveness returns HTTP 200 and the exact selected SHA;
- database-backed readiness returns HTTP 200 and the exact selected SHA;
- production security headers retain `X-Content-Type-Options: nosniff`;
- the configured stable Web origin receives credentialed CORS authority;
- unauthenticated `/auth/me` fails closed with HTTP 401;
- production-shaped Swagger/API documentation is not publicly reachable.

Negative hostile-origin CORS behavior remains repository-qualified. The live smoke does not manufacture a production 5xx solely to exercise that rejection path.

This smoke is a release-integrity check, not the complete product journey suite. The richer staging/live black-box journeys remain tracked by the public-beta gate.

## Privacy-safe release receipts

Successful staging and production deployments retain schema-v2 JSON receipts.

Receipts contain only operational metadata:

- environment;
- exact release SHA;
- repository/workflow/run identity;
- API HTTPS origin;
- stable Web HTTPS origin;
- immutable Web deployment HTTPS URL;
- booleans proving main ancestry, canonical exact-SHA checks, API/Web release identity, Web/API provenance, live black-box smoke, and staging qualification where applicable.

They do not contain credentials, bearer tokens, request bodies, user/pet identifiers, database rows, free-form user content, or provider payloads.

Staging receipts are retained for 30 days. Production receipts are retained for 90 days.

A receipt cannot be created unless canonical main checks and live smoke have passed. A production receipt additionally requires successful exact-SHA staging qualification.

## External configuration required

Repository code cannot manufacture provider authority.

### GitHub `staging` environment

Secrets:

- `FLY_API_TOKEN`
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Variable:

- `WEB_ORIGIN`

### GitHub `production` environment

Secrets:

- `FLY_API_TOKEN`
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`
- optional `SLACK_WEBHOOK`

Variable:

- `WEB_ORIGIN`

The production environment should require explicit reviewer approval once repository/environment administration is configured.

### Vercel

Two intended Woof Web projects must exist. The staging and production environment IDs must refer to the correct isolated projects. Do not reuse an unrelated project merely to satisfy the credential gate.

### Fly.io

Staging and production deploy tokens must be authorized for the intended `woof-api-staging` and `woof-api-prod` applications respectively. Runtime application secrets such as database, JWT, CORS, metrics, storage, and optional provider credentials belong to the corresponding Fly application, not source control.

### Database

Staging and production use isolated managed PostgreSQL authority. The production provider must expose real backup/snapshot authority suitable for the restore rehearsal in issue #100.

## Repository governance boundary

This release authority does not replace branch protection.

`main` still requires the GitHub ruleset tracked by issue #80. The minimum source boundary is pull-request-only ordinary changes, required unconditional release checks, no force pushes/deletion, current-branch/merge-queue semantics, resolved conversations, and minimal explicit bypass.

Until that external admin control is enabled, Woof must not claim complete production source-governance authority.

## Recovery boundary

A deployable release is not yet a recoverable service.

Before public beta production promotion, issue #100 Phase 4 requires a real backup/restore rehearsal against the selected database provider. After production deployment, Phase 6 requires live monitoring/error/metrics/alert evidence tied to the deployed SHA.

Do not interpret “provider says backups are enabled” as restore proof and do not describe repository alert configuration as proof that a live operator receives alerts.

## Rollback

Application rollback is an explicit new production promotion of a previously staging-qualified exact `main` SHA. The rollback candidate must satisfy the same production preflight, including canonical exact-SHA checks and successful staging qualification.

Database rollback is not implied by application rollback. Migrations remain forward-safe authority. Destructive schema rollback or backup restoration requires its own reviewed recovery procedure.

## Qualification

Repository qualification is owned by:

```bash
python3 .github/scripts/assert-release-promotion-authority.py
python3 .github/scripts/verify-main-release-checks.py --self-test
node .github/scripts/write-release-receipt.mjs --self-test
node .github/scripts/verify-web-deployment-provenance.mjs --self-test
node .github/scripts/verify-live-release.mjs --self-test
python3 .github/scripts/assert-operational-privacy-release.py
```

These checks run in the always-on `Release Promotion Authority CI` lane.

## Evidence boundary

Passing repository CI proves the promotion contract and receipt machinery. It does **not** prove that Fly, Vercel, GitHub environments, database backups, alert routing, physical-device distribution, or real-user pilots have been configured or exercised.

Those remain separate launch gates and must only be claimed after live evidence exists.
