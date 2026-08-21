# Woof Adventure release qualification

This document is the merge checkpoint for the Adventure System integration branch. The release is not merge-ready unless the exact PR head, and later the exact rebased merge candidate, satisfy these gates.

## Repository gates

- Prisma schema validates and remains formatter-clean.
- The complete PostgreSQL migration chain applies to a fresh pgvector/PostgreSQL 15 database.
- The Adventure migration remains additive and contains no destructive `DROP`, `TRUNCATE`, or bulk `DELETE` operations.
- The freshly migrated database matches the Prisma datamodel according to `prisma migrate diff`.
- Prettier passes on all branch-owned Adventure, CareEvent, Activity/Coach integration, Pack, Compass, Journey, navigation, API-client, and compatibility-session surfaces.
- ESLint runs with zero warnings on the release-critical API and web surfaces touched by this stack.
- The complete API TypeScript project type-checks.
- The complete web TypeScript project must type-check before the stack is considered production-ready. If inherited web debt remains, it is a named stack blocker rather than something hidden with `ignoreBuildErrors`.
- Bond XP policy contract tests pass.
- Real PostgreSQL reward-ledger integrity and concurrency tests pass.
- Quest-engine safety, media-ownership, snapshot, and telemetry-failure tests pass.
- The full API Jest suite passes.
- Production API and web builds pass.

## Reward and safety invariants

1. Clients never submit arbitrary XP values.
2. A CareEvent dedupe key cannot mint reward twice for the same user.
3. Concurrent requests for one user are serialized before cap and dedupe calculation.
4. Anti-farming windows use trusted server issuance time rather than client-controlled occurrence timestamps.
5. Zero-XP and safety-ineligible events do not degrade later legitimate rewards.
6. Recovery can earn legitimate progress.
7. A safe opt-out can earn Bond progress.
8. A stressful or `not_their_thing` result does not inflate the original social/training pathway merely because a quest was closed.
9. Optional media has only a tiny reward effect, and a memory bonus is accepted only for an owned `READY` media asset belonging to the pet.
10. Raw distance, calories, duration, money spent, and photo volume do not multiply Bond XP without bound.
11. Selected quests can be completed after a same-day deck reshuffle using a short-lived semantic snapshot, but current server RewardPolicy remains the only XP authority.
12. The Pawprint Compass is opportunity coverage, never a health or ownership score.
13. Pack challenges are cooperative aggregates, not raw-volume rankings.
14. `emergency_now` Health Lens results suppress the global game navigation and never issue XP, confetti, or quest-completion treatment.
15. Adventure and Pack public routes can be disabled at runtime with `ENABLE_ADVENTURE_SYSTEM=false` in production.

## Stack rule

PR #6 is stacked on `agent/pet-media-library-v1`. Green CI on the stacked base is necessary but not sufficient for final merge. After the parent stack lands on `main`, rebase or restack the Adventure branch onto the resulting `main` and run the complete qualification pipeline again on that exact candidate.

## Release evidence

Do not mark the PR ready because an earlier commit was green. Record the final commit SHA and corresponding successful Adventure System CI run in the PR description before changing draft status.
