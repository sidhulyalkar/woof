# CI Action Runtime Contract

Woof keeps the runtime used by GitHub Actions themselves separate from the Node.js runtime used to build and test the product.

## Maintained action executors

Repository workflows standardize on these action majors:

- `actions/checkout@v7`
- `actions/setup-node@v7`
- `pnpm/action-setup@v6`
- `actions/upload-artifact@v7`

These maintained action lines use the current GitHub Actions Node 24 execution generation instead of the deprecated Node 20 action runtime that GitHub-hosted runners were already forcing forward during qualification runs.

The repository does not use removed `actions/setup-node` inputs such as `always-auth`, and it does not depend on an `actions/download-artifact@v4` workflow contract.

The Fly CLI setup action is additionally pinned to immutable upstream commit `fc53c09e1bc3be6f54706524e3b82c4f462f77be`, while the installed `flyctl` binary remains pinned to `0.4.76`. Deployment workflows must not float this action independently of the qualification lane.

## Product toolchain remains unchanged

This contract does **not** migrate the application runtime.

Woof continues to build and test with:

- Node.js `20.20.2`
- pnpm `8.15.1`

`actions/setup-node@v7` installs that exact project Node version. `pnpm/action-setup@v6` installs that exact pnpm version.

## Deployment credential authority

Staging and production deployment workflows fail before deployment work when their required deployment identity is absent.

The API deployment requires:

- `FLY_API_TOKEN`

The Web deployment requires:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Those values belong in the corresponding GitHub environment or repository secret store. The workflow does not invent fallback credentials, run an interactive login, or silently create/link a Vercel project when release identity is unknown.

Production Slack notification is optional. `SLACK_WEBHOOK` may be omitted without changing deployment success. The notification path uses a small `curl` webhook request and is `continue-on-error`; an unavailable notification sink cannot turn a successful API/Web release into a failed deployment. The legacy `8398a7/action-slack` integration is forbidden because its old input/runtime contract both drifted from the workflow and obscured the actual deployment result.

## Executable drift guard

`.github/workflows/ci-action-runtime-ci.yml` runs for workflow changes and:

1. uses the maintained checkout, pnpm, setup-node, and artifact-upload actions itself;
2. proves the project runtime resolves to Node `20.20.2` and pnpm `8.15.1`;
3. scans every workflow and rejects an unapproved major for the standardized action families or an unapproved Fly setup ref;
4. checks staging and production for explicit Fly/Vercel credential preflight contracts;
5. rejects the legacy Slack action and verifies notification failure cannot own production release success;
6. runs Prettier over the complete workflow inventory so malformed YAML cannot pass silently;
7. performs a frozen lockfile install; and
8. uploads a one-day qualification artifact so `actions/upload-artifact@v7` is exercised on a successful path.

A future action-major, deployment-identity, or notification-contract change should be an explicit, qualified infrastructure release rather than silent version drift.

## Release qualification

Changing shared workflow infrastructure has a wide ownership surface. A release is qualified only when every workflow actually triggered by the PR completes successfully on one frozen exact head SHA. Earlier diagnostic heads do not count.

Passing CI proves the deployment workflow is structurally executable. It does **not** prove external deployment credentials exist in a GitHub environment. A real push/dispatch deployment remains the final integration proof, including Fly deployment, API liveness/readiness, and the Vercel production release.
