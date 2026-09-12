# Release Promotion Authority v1

## Purpose

Woof treats repository qualification, staging qualification, and production promotion as different claims.

A green pull request does not mean a release is deployed. A merge to `main` does not mean a release is production. Production is an explicit promotion of one exact commit that has already completed the staging deployment workflow successfully.

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
    +--> Deploy to Staging
    |      - exact SHA validation
    |      - main ancestry validation
    |      - API migration/deploy
    |      - live + ready exact-SHA verification
    |      - Web deploy
    |      - Web release/API-origin provenance verification
    |      - privacy-safe staging release receipt
    |
    v
explicit workflow_dispatch(release_sha)
    |
    v
Deploy to Production
       - exact SHA validation
       - main ancestry validation
       - successful exact-SHA staging workflow required
       - production environment authority
       - API migration/deploy
       - live + ready exact-SHA verification
       - Web deploy
       - Web release/API-origin provenance verification
       - privacy-safe production release receipt
```

## Staging

`Deploy to Staging` runs automatically for pushes to `main`. A manual dispatch re-runs the workflow for the Git ref selected when dispatching; the workflow still rejects that candidate unless its exact SHA is reachable from `origin/main`.

The staging workflow rejects:

- branch names, abbreviated SHAs, uppercase/non-hex release identifiers, and other ambiguous values as release identity;
- commits that are not reachable from `origin/main`;
- a checkout whose resolved `HEAD` differs from the workflow's exact candidate SHA;
- missing deployment credentials;
- API liveness/readiness responses that do not report the exact candidate SHA;
- Web deployments whose embedded release identity or API origin does not match the expected release.

The old `develop`-branch trigger was removed because no canonical `develop` branch exists. A dormant staging workflow is not staging authority.

## Production

`Deploy to Production` is manual-only. It does not run on a push to `main`.

The operator must provide the exact 40-character release SHA. Before any production deployment credentials are used, the workflow proves:

1. the requested SHA resolves exactly;
2. it is reachable from `origin/main`;
3. GitHub Actions has a successful `Deploy to Staging` workflow run for that exact SHA.

Only after those checks pass can the production environment jobs start.

This allows repository environment protection and required reviewers to remain an additional external control without making a merge itself equivalent to a production release.

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

- API `/ops/health/live` and `/ops/health/ready` must report the same exact SHA;
- the Web deployment must expose the expected `woof-release` identity;
- the Web deployment must expose the expected `woof-api-origin` identity.

A successful workflow therefore means the released artifacts were checked against the selected candidate, not merely against whatever commit happened to trigger the workflow.

## Privacy-safe release receipts

Successful staging and production deployments retain small JSON receipts as GitHub Actions artifacts.

Receipts contain only operational metadata:

- environment;
- exact release SHA;
- repository/workflow/run identity;
- API and Web HTTPS origins;
- booleans recording the release/provenance checks that had to pass before the receipt job could run.

They do not contain credentials, bearer tokens, request bodies, user/pet identifiers, database rows, free-form user content, or provider payloads.

Staging receipts are retained for 30 days. Production receipts are retained for 90 days.

A production receipt cannot be generated unless exact-SHA staging qualification was verified by the production workflow.

## External configuration still required

This repository code cannot manufacture provider authority. Before the first live release, operators must configure:

### GitHub `staging` environment

- `FLY_API_TOKEN`
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

### GitHub `production` environment

- `FLY_API_TOKEN`
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`
- optional `SLACK_WEBHOOK`

The production environment should require explicit reviewer approval once repository/environment administration is configured.

### Vercel

An intended Woof Web project must exist and the IDs above must refer to that project. Do not reuse an unrelated Vercel project merely to satisfy the credential gate.

### Fly.io

The staging and production tokens must be authorized for the intended `woof-api-staging` and `woof-api-prod` applications respectively.

## Repository governance boundary

This release authority does not replace branch protection.

`main` still requires a GitHub ruleset/branch-protection boundary that, at minimum:

- requires pull requests for ordinary changes;
- requires the intentionally selected release-critical checks;
- prevents force pushes and branch deletion;
- keeps bypass authority minimal and explicit.

Until that external repository-admin control is enabled, Woof must not claim complete production source-governance authority.

## Rollback

Rollback is an explicit new production promotion of a previously staging-qualified exact `main` SHA. Do not mutate release identity or redeploy an unknown local checkout.

The chosen rollback SHA must satisfy the same production preflight, including a successful `Deploy to Staging` workflow run for that exact SHA. Commits from before this promotion authority existed are not grandfathered into production eligibility merely because they once existed on `main`.

Database rollback is not implied by application rollback. Migrations must remain forward-safe, and destructive schema rollback requires its own reviewed recovery procedure.

## Qualification

Repository qualification is owned by:

```bash
python3 .github/scripts/assert-release-promotion-authority.py
node .github/scripts/write-release-receipt.mjs --self-test
node .github/scripts/verify-web-deployment-provenance.mjs --self-test
python3 .github/scripts/assert-operational-privacy-release.py
```

These checks run in `Release Promotion Authority CI`.

## Evidence boundary

Passing this CI proves the repository's promotion contract and receipt machinery. It does **not** prove that Fly, Vercel, GitHub environments, alert routing, backups, physical-device distribution, or real-user pilots have been configured or exercised.

Those remain separate launch gates and must only be claimed after live evidence exists.
