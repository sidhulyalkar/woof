# Woof Adventure release qualification

This file exists as the human-authored qualification checkpoint for the Adventure System integration branch.

The release is not considered merge-ready unless the exact PR head passes the following repository gates:

- complete PostgreSQL migration chain, including `20260821223000_add_woof_adventure_system`;
- Prettier on every Adventure, CareEvent, Activity, Coach-integration, Pack, Compass, Journey, and navigation surface;
- monorepo ESLint;
- monorepo TypeScript type-check;
- Bond XP policy contract tests;
- backend unit/API tests;
- production API build;
- production web build.

Product invariants that must remain true after future changes:

1. Clients never submit arbitrary XP values.
2. A CareEvent dedupe key cannot mint reward twice for the same user.
3. Recovery can earn legitimate progress.
4. A safe opt-out can earn Bond progress.
5. A stressful or `not_their_thing` result does not inflate the original social/training pathway just because a quest was closed.
6. Optional media has only a tiny reward effect.
7. Raw distance, calories, and duration do not multiply Bond XP without bound.
8. The Pawprint Compass is opportunity coverage, never a health or ownership score.
9. Pack challenges are cooperative aggregates, not raw-volume rankings.
10. `emergency_now` Health Lens results suppress the global game navigation and never issue XP, confetti, or quest-completion treatment.
