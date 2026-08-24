# dogOS Production Runtime Hardening v1

This release hardens the Woof API and its deployment path without changing product authority, database schema, or user-facing feature semantics.

## Runtime privacy

Production exception telemetry is deliberately low-cardinality and identifier-free.

For server failures, operational context may include only:

- a bounded `X-Request-ID` correlation value;
- HTTP method;
- HTTP status;
- the framework route template, such as `/api/v1/users/:userId`.

The exception boundary must not export request headers, cookies, authorization tokens, query strings, request bodies, user IDs, emails, raw request URLs, breadcrumb payloads, SQL text, provider URLs, or span data to Sentry.

Client errors are returned to the caller but are not reported to Sentry. HTTP 429 may be logged as a low-cardinality warning; other expected 4xx responses do not become production error logs.

Error response paths strip the query string. Every request receives an `X-Request-ID` response header. A caller-supplied request ID is accepted only when it satisfies the bounded single-line format; otherwise the server generates a UUID.

## Runtime lifecycle

The Nest application enables shutdown hooks for `SIGTERM` and `SIGINT`. This allows module teardown, including Prisma disconnect, to run during orchestrated shutdown.

Bootstrap failures are caught at the process boundary, logged without arbitrary secret-bearing error text, and set a non-zero process exit code.

Swagger is fail-closed. `API_DOCS_ENABLED=false` is the default and `/docs` is not mounted unless the environment explicitly opts in.

## Container contract

`infra/docker/Dockerfile.api` is the production image source.

The image:

- pins Node `20.20.2` and pnpm `8.15.1`;
- installs that exact pnpm version during image construction so release commands do not download package-manager code at runtime;
- installs OpenSSL and Alpine compatibility libraries required by the Prisma native runtime;
- builds from the committed lockfile;
- generates the committed Prisma client before the API build;
- runs the final application and migration release command as the unprivileged `node` user;
- gives that runtime user ownership of the copied workspace needed by the Prisma migration engine, rather than escalating the container back to root;
- enables Node source maps for actionable stack traces;
- uses `/api/v1/ops/health/live` for container liveness.

The runtime image intentionally keeps the database workspace and installed Prisma CLI for this release so the deployment release command can apply committed migrations. Image-size optimization is a separate concern and must not remove migration capability accidentally.

The non-root migration contract is tested because Prisma may need package-local native-engine runtime state while resolving its migration binary. A production image that builds successfully but cannot execute `prisma migrate deploy` as its configured user is considered invalid.

`.dockerignore` excludes local dependencies, build outputs, reports, logs, Git metadata, and environment files from the remote Docker build context.

## Fly deployment contract

`apps/api/fly.toml` is shared by staging and production. The app name is supplied by the GitHub workflow, so one checked-in runtime contract cannot drift between environments.

Deployment invariants:

- committed Prisma migrations run once as the Fly release command before rollout;
- a failed migration aborts deployment;
- rollout uses the readiness endpoint `/api/v1/ops/health/ready`;
- HTTPS is forced;
- production-style Machines do not auto-stop to zero;
- at least one Machine remains warm;
- graceful termination uses `SIGTERM` with a bounded shutdown window;
- API documentation stays disabled by default.

Liveness answers whether the process is alive. Readiness answers whether it is safe to receive traffic. They are intentionally separate.

## GitHub deployment contract

Both staging and production workflows:

- use a pinned Fly setup action and pinned flyctl version;
- deploy from the repository root with the checked-in Fly configuration;
- rely on release migrations and readiness gates rather than a fixed sleep;
- verify both liveness and readiness with bounded HTTP retries;
- use a pinned Vercel CLI instead of `latest`.

The web deployment waits for the API deployment in both environments, preventing a frontend release from advertising an API rollout that failed its migration or readiness gate.

## Qualification

The dedicated Production Runtime CI lane must remain read-only and prove one exact candidate head. It must:

1. reject database schema or migration ownership in this release;
2. apply the full existing migration chain to real PostgreSQL;
3. pass exact formatting, zero-warning lint, and API TypeScript checks;
4. enforce telemetry privacy, request-correlation, graceful-shutdown, docs-gating, and deployment-configuration invariants;
5. run focused runtime privacy and observability tests;
6. build the API;
7. build the production Dockerfile from the sanitized repository context;
8. execute `pnpm --filter @woof/database db:migrate:deploy` inside the built production image as its configured non-root user and without a runtime package-manager fetch;
9. boot that image against PostgreSQL as the configured runtime user;
10. verify liveness, readiness, default-disabled Swagger, and `X-Request-ID` over real HTTP.

Diagnostic heads do not qualify a release.
