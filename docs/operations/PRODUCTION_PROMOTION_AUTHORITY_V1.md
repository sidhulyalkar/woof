# Production Promotion Authority v1

Woof production deployment is a promotion decision, not a side effect of merging code.

This contract separates four different statements:

1. a commit exists on canonical `main`;
2. that commit has repository qualification evidence;
3. that commit has been deployed and qualified in staging;
4. an operator deliberately promotes that exact commit to production.

Green CI is not a production-deployment claim, and a production deployment is not a product-validation claim.

## Canonical flow

```text
pull request
    |
    v
qualified main commit
    |
    v
automatic staging deploy
    |
    v
staging live-release receipt
    |
    v
production verifies exact staging receipt + workflow run
    |
    v
operator selects exact 40-hex main SHA
    |
    v
production environment approval / authority
    |
    v
exact-SHA API + Web production deploy
    |
    v
production live-release receipt
```

## Staging authority

`.github/workflows/deploy-staging.yml` runs after a push to `main`.

The workflow:

- deploys the API with `WOOF_RELEASE_SHA` equal to the exact `github.sha`;
- requires API liveness and database-backed readiness to report that same release;
- builds Web with the same release identity and the canonical staging API URL;
- verifies both the transient Vercel deployment and the stable public Web origin expose that exact release and API provenance;
- verifies the configured public Web origin is actually admitted by API CORS;
- writes a privacy-safe `woof-live-release-qualification` receipt; and
- retains that receipt as `staging-release-<sha>`.

Staging therefore answers: **did the exact merged commit become a coherent live Web/API release at the public staging boundary?**

It does not prove production, device distribution, recovery, or user value.

## Production authority

`.github/workflows/deploy-production.yml` has no `push` trigger. It is manually dispatched with one required `release_sha`.

Before deployment, the workflow rejects the request unless:

- the workflow itself was dispatched from `main`;
- `release_sha` is one exact lowercase 40-hex Git SHA;
- the commit exists in repository history;
- the commit is an ancestor of the current canonical `origin/main`;
- a non-expired `staging-release-<sha>` artifact exists for that exact SHA and `main` branch;
- the artifact came from a successful `push` run of `.github/workflows/deploy-staging.yml` at that exact SHA; and
- the downloaded receipt itself reports staging, the exact release, live/ready/database-ready API state, exact transient + stable Web provenance, intended API origin, and credentialed CORS qualification.

The API and Web jobs then check out that exact SHA. Release identity is derived from `release_sha`, never from the workflow-dispatch commit.

This prevents a later `main` commit from silently becoming the deployed artifact and prevents an operator from promoting a canonical commit that never completed the staging live-release gate.

## Environment authority required outside the repository

Both `staging` and `production` GitHub environments require externally configured deployment authority.

Secrets:

- `FLY_API_TOKEN`
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Optional notification secret:

- `SLACK_WEBHOOK`

Public environment variable:

- `WEB_ORIGIN` — one stable HTTPS origin only, for example `https://staging.example.com` or `https://www.example.com`.

`WEB_ORIGIN` is not inferred from a transient Vercel deployment URL. It represents the stable public alias that must expose the selected release and that the API must admit through CORS.

Production should also use GitHub environment protection / reviewers where available. Repository code can require the `production` environment, but repository code cannot prove that an administrator configured reviewers or secret scope correctly.

The production workflow has read-only Actions permission in addition to repository-content read permission solely so it can authenticate the exact staging workflow run and download its retained receipt. It does not gain Actions write authority.

## Live qualification receipt

`.github/scripts/qualify-live-release.mjs` performs non-destructive black-box release checks against public endpoints.

It verifies:

- API `/ops/health/live` returns `status=live` and the exact expected release;
- API `/ops/health/ready` returns `status=ready`, database `status=ready`, and the same release;
- the API admits the configured public Web origin through credentialed CORS;
- the transient Vercel deployment `/demo` page exposes the exact release SHA and intended API base URL; and
- the stable public Web origin `/demo` page exposes the same exact release SHA and intended API base URL.

This deliberately catches the case where deployment succeeds but the stable alias still points at an older artifact.

The retained receipt contains only low-cardinality public operational facts:

- environment class;
- qualification timestamp;
- exact release SHA;
- public API base URL;
- transient Web deployment origin;
- stable public Web origin;
- health status classes;
- CORS authority result; and
- names of checks performed.

It intentionally excludes user IDs, pet IDs, household IDs, emails, tokens, request bodies, provider payloads, free-form notes, database rows, and raw response bodies.

## What v1 does not yet prove

This tranche deliberately does not pretend to solve the remaining launch gates.

Still required before public beta:

- repository ruleset / protected `main` configured administratively;
- an actually successful staging deployment with retained receipt;
- deliberate promotion of the same selected release to production;
- authenticated synthetic black-box journeys against the live stack;
- realtime revoke/block/logout authority checked live;
- backup and restore rehearsal;
- live error-monitoring and alert-routing drill;
- physical iOS/TestFlight qualification and accessibility evidence;
- structured coarse-region Pack authority; and
- a real owner/advisor pilot.

## Promotion procedure

1. Merge only a fully qualified release candidate to `main`.
2. Confirm `Deploy to Staging` succeeds for that exact SHA.
3. Inspect the retained `staging-release-<sha>` receipt and confirm both deployment and stable public origins report the same SHA.
4. Resolve any staging, alias, CORS, or provider discrepancy before production promotion.
5. Dispatch `Deploy to Production` from `main` and paste the exact staged 40-hex SHA into `release_sha`.
6. The workflow independently locates the exact staging artifact, verifies its successful staging workflow run, downloads it, and validates its contents before any production deploy job can start.
7. Complete any configured production-environment approval.
8. Confirm API migration, liveness, readiness, deployment provenance, stable-origin provenance, CORS, and live qualification succeed.
9. Retain and inspect `production-release-<sha>`.
10. Continue with authenticated live black-box and operational drills. Do not treat the receipt alone as full public-beta evidence.

## Rollback boundary

A previous known-good `main` SHA can be deliberately re-promoted through the same production workflow because the selector accepts canonical `main` ancestors, but it still needs a valid retained staging receipt for that exact SHA.

That is application rollback authority, not database rollback authority. Database migrations must remain forward-safe or use an explicitly rehearsed recovery procedure. Never infer that deploying an older application SHA reverses a migrated production schema.
