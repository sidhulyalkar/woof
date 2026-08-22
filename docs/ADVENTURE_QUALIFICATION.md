# Woof Adventure release qualification

This document is the merge checkpoint for the Adventure System integration branch. The release is not merge-ready unless the exact PR head, and later the exact rebased merge candidate, satisfy these gates.

## Repository gates

- Prisma schema validates and remains formatter-clean.
- The complete PostgreSQL migration chain applies to a fresh pgvector/PostgreSQL 15 database.
- The current chain includes a forward-only normalization of the inherited Media Library index name so the physical database and Prisma datamodel can remain zero-diff.
- The Adventure migration remains additive and contains no destructive `DROP`, `TRUNCATE`, or bulk `DELETE` operations.
- The freshly migrated database matches the Prisma datamodel according to `prisma migrate diff`.
- Prettier passes on all branch-owned Adventure, CareEvent, operational-health, Activity/Coach integration, Pack, Compass, Journey, navigation, API-client, onboarding, and repaired compatibility surfaces.
- ESLint runs with zero warnings on the release-critical API and web surfaces touched by this stack.
- The complete API TypeScript project type-checks.
- The complete web TypeScript project type-checks. Type failures are blockers rather than something hidden with `ignoreBuildErrors`.
- Adventure web transport contract tests pass and prove the browser never sends client-controlled XP in quest completion requests.
- Bond XP policy contract tests pass.
- Real PostgreSQL reward-ledger integrity and concurrency tests pass.
- Quest-engine safety, media-ownership, snapshot, and telemetry-failure tests pass.
- The full API Jest suite passes.
- Production API and web builds pass.
- API liveness remains dependency-free, while readiness performs a real database probe and fails closed with HTTP 503 if PostgreSQL is unavailable.
- Production Swagger is disabled unless `API_DOCS_ENABLED=true` is explicitly configured.

## Reward and safety invariants

1. Clients never submit arbitrary XP values.
2. A CareEvent dedupe key cannot mint reward twice for the same user.
3. Concurrent requests for one user are serialized before cap and dedupe calculation.
4. Anti-farming windows use trusted server issuance time rather than client-controlled occurrence timestamps.
5. A future or invalid occurrence timestamp is normalized to server time before persistence, so it cannot make Compass or Rhythm appear artificially recent.
6. Legitimate historical/offline occurrence timestamps are preserved.
7. Zero-XP and safety-ineligible events do not degrade later legitimate rewards.
8. Recovery can earn legitimate progress.
9. A safe opt-out can earn Bond progress.
10. A stressful or `not_their_thing` result does not inflate the original social/training pathway merely because a quest was closed.
11. Optional media has only a tiny reward effect, and a memory bonus is accepted only for an owned `READY` media asset belonging to the pet.
12. Raw distance, calories, duration, money spent, and photo volume do not multiply Bond XP without bound.
13. Selected quests can be completed after a same-day deck reshuffle using a short-lived semantic snapshot, but current server RewardPolicy remains the only XP authority.
14. The Pawprint Compass is opportunity coverage, never a health or ownership score.
15. Pack challenges are cooperative aggregates, not raw-volume rankings.
16. `emergency_now` Health Lens results suppress the global game navigation and never issue XP, confetti, or quest-completion treatment.
17. Adventure and Pack public routes can be disabled at runtime with `ENABLE_ADVENTURE_SYSTEM=false` in production.
18. Signed private-media upload URLs are returned ephemerally and are never persisted in Media Library records.

## Operational gates

Before production traffic is enabled:

1. Take or verify a recoverable production database backup/snapshot.
2. Apply migrations with `prisma migrate deploy`; never use an interactive development migration command against production.
3. Verify `/api/v1/health/live` returns 200.
4. Verify `/api/v1/health/ready` returns 200 and reports the database check as `up`.
5. Deploy application code with `ENABLE_ADVENTURE_SYSTEM=false` first.
6. Smoke-test authentication, pet ownership boundaries, activity persistence, Health Lens emergency suppression, private Media Library access, and Adventure-disabled behavior.
7. Enable Adventure deliberately only after the database and application candidate are proven healthy.
8. Observe reward-decision and application-error telemetry during rollout.
9. Roll application behavior back by disabling Adventure/reverting code if necessary; preserve additive Adventure tables and immutable reward history unless a separate approved data-removal procedure exists.

## Stack rule

PR #6 is stacked on `agent/pet-media-library-v1`. Green CI on the stacked base is necessary but not sufficient for final merge. After the parent stack lands on `main`, rebase or restack the Adventure branch onto the resulting `main` and run the complete qualification pipeline again on that exact candidate.

## Evidence policy

A successful earlier commit is useful as a checkpoint but is not final merge evidence. Run #179 established the first fully green baseline across migrations, zero-diff Prisma parity, formatting/lint, both TypeScript projects, reward policy, real-PostgreSQL concurrency, Quest contracts, the complete API suite, and both production builds. Subsequent operational-hardening commits must pass the expanded workflow again.

Do not mark the PR ready because an earlier commit was green. Record the final commit SHA and corresponding successful Adventure System CI run in the PR description before changing draft status.
