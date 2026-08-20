# Contributing to Woof

Woof is organized as a product monorepo. Contributions should improve the canonical system under `apps/`, `packages/`, `ml/`, `infra/` or `n8n/`; code under `legacy/` is preserved history and should not receive new product work.

## Development principles

1. **Keep product truth deterministic.** Optional ML, notification, analytics or storage services must not silently corrupt canonical state when unavailable.
2. **Prefer one contract per domain.** Avoid creating parallel client and API representations that drift apart.
3. **Make recommendations explainable.** New ranking logic should expose provenance, confidence and interpretable factors where possible.
4. **Treat location as sensitive.** Never add public precise coordinates or route history for convenience.
5. **Test authorization, not only happy paths.** User-owned resources and realtime rooms need explicit access-control coverage.
6. **Separate evidence from aspiration.** Documentation should label experimental or planned work rather than presenting it as integrated behavior.
7. **Preserve accessibility.** Keyboard navigation, visible focus, semantic controls, reduced motion and usable touch targets are baseline requirements.

## Local setup

```bash
pnpm install
docker compose up -d
cp apps/api/.env.example apps/api/.env
cp apps/web/.env.local.example apps/web/.env.local
pnpm --filter @woof/database db:generate
pnpm --filter @woof/database db:migrate
pnpm --filter @woof/api db:seed
```

Start the API and web client in separate terminals:

```bash
pnpm --filter @woof/api dev
pnpm --filter @woof/web dev
```

## Before opening a change

Run the repository verification gates:

```bash
pnpm format:check
pnpm lint
pnpm type-check
pnpm test
pnpm build
```

For browser-facing changes, also run:

```bash
pnpm --filter @woof/web test:e2e
```

## Pull request expectations

A strong change explains:

- the user or system problem being solved,
- the chosen approach and important tradeoffs,
- how failure/degraded behavior works,
- tests or other evidence used to validate it,
- privacy/security implications when relevant,
- screenshots or recordings for meaningful visual changes.

Avoid mixing unrelated cleanup, large dependency upgrades and product changes in one review when they can be separated.

## ML changes

Read [`docs/ML_SYSTEM.md`](docs/ML_SYSTEM.md) before promoting a model or changing compatibility behavior. Training successfully is not sufficient evidence for product promotion. At minimum, compare against the deterministic baseline on a leakage-resistant split and preserve score provenance through serving.

## Documentation

Canonical portfolio documentation lives in `README.md` and `docs/`. Historical implementation reports are intentionally archived under `docs/archive/` and should not be updated to describe current system status.
