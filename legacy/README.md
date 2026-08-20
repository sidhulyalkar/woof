# Legacy Prototypes

The directories here are preserved snapshots from before Woof converged on its current monorepo architecture.

They are **not active application entry points** and should not be used for new development or deployment.

## Preserved prototypes

- `backend-fastapi-prototype/` — early sample FastAPI backend with mocked endpoints and the historical PetPath identity.
- `vercel-mock-frontend/` — early Next.js mock-data frontend used to explore product surfaces before the canonical API integration.

## Canonical applications

Current development happens in:

- [`../apps/web`](../apps/web) — Next.js web/PWA client
- [`../apps/mobile`](../apps/mobile) — Expo React Native client
- [`../apps/api`](../apps/api) — NestJS application API
- [`../ml`](../ml) — experimental Python model/inference work

The prototypes remain here to show the project’s evolution without creating ambiguity about which code runs the current product.
