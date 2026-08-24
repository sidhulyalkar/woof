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

## Product toolchain remains unchanged

This release does **not** migrate the application runtime.

Woof continues to build and test with:

- Node.js `20.20.2`
- pnpm `8.15.1`

`actions/setup-node@v7` installs that exact project Node version. `pnpm/action-setup@v6` installs that exact pnpm version.

The release also leaves Fly CLI, Vercel CLI, Docker image Node, database migrations, dependency lockfiles, and deployment behavior unchanged.

## Executable drift guard

`.github/workflows/ci-action-runtime-ci.yml` runs for workflow changes and:

1. uses the maintained checkout, pnpm, setup-node, and artifact-upload actions itself;
2. proves the project runtime resolves to Node `20.20.2` and pnpm `8.15.1`;
3. scans every workflow and rejects an unapproved major for the four standardized action families;
4. runs Prettier over the complete workflow inventory so malformed YAML cannot pass silently;
5. performs a frozen lockfile install; and
6. uploads a one-day qualification artifact so `actions/upload-artifact@v7` is exercised on a successful path.

A future action-major change should be an explicit, qualified infrastructure release rather than silent version drift.

## Release qualification

Changing shared workflow infrastructure has a wide ownership surface. A release is qualified only when every workflow actually triggered by the PR completes successfully on one frozen exact head SHA. Earlier diagnostic heads do not count.
