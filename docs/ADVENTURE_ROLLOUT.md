# Adventure System rollout and rollback

The Adventure System is designed for a staged launch. The database migration is additive, while the user-facing API can be disabled at runtime.

## Production defaults

- Keep `ENABLE_ADVENTURE_SYSTEM=false` until the migration and application version are deployed and smoke-tested.
- Production must opt in explicitly with `ENABLE_ADVENTURE_SYSTEM=true`.
- Development and CI may enable Adventure so the release path is exercised continuously.
- The migration should be deployed before enabling Adventure routes.

## Dark launch

Before exposing the redesign broadly:

1. deploy the additive migration;
2. deploy the API and web build with the public Adventure feature flag disabled;
3. verify API health, migration status, and error rate;
4. enable Adventure in a staging or limited environment;
5. exercise `GET /adventure/me`, quest selection, completion, duplicate completion, safe opt-out, Activity emission, Coach emission, Pack aggregation, Journey, and Health emergency suppression;
6. verify one completion produces one CareEvent and one RewardLedger row;
7. issue concurrent duplicate completions and confirm only one reward receipt is minted;
8. confirm reward decision logs contain event semantics and receipt identifiers but no raw user or pet identifiers;
9. inspect daily/pathway cap behavior and duplicate rate for unexpected patterns;
10. enable production only after the exact release candidate has passed `Adventure System CI`.

## Operational signals

Watch at minimum:

- Adventure endpoint 5xx rate;
- database transaction/lock latency around reward issuance;
- duplicate reward-request rate;
- zero-XP rate and cap-hit rate by event type/pathway;
- CareEvent rows without a RewardLedger row;
- RewardLedger rows without a CareEvent row;
- Activity/Coach reward-emission failures, which must never make the underlying saved action fail;
- quest acceptance and completion without using screen time as a success target;
- safe-opt-out frequency as a legitimate welfare signal, not a failure metric.

The north-star product metric remains Weekly Meaningful Dyad Actions, not clicks, streak pressure, miles, or posting volume.

## Incident rollback

The safest rollback preserves data.

1. Set `ENABLE_ADVENTURE_SYSTEM=false` to make public Adventure and Pack routes unavailable without deleting ledger history.
2. If application code still causes a regression outside guarded routes, revert the application commit/PR and redeploy.
3. Leave the additive Adventure tables in place during ordinary rollback. They are backward-compatible with older application code and contain useful immutable reward history.
4. Do **not** automatically generate or execute a destructive down migration during an incident. Dropping `care_events`, `reward_ledger`, or `quest_interactions` destroys evidence needed to audit or recover the launch.
5. If schema removal is ever required, treat it as a separate reviewed data-retention operation with an explicit backup and migration plan.

## Parent-stack integration

PR #6 currently targets `agent/pet-media-library-v1`. After that parent line is integrated into `main`:

```bash
git fetch origin main
git rebase --onto origin/main agent/pet-media-library-v1 agent/woof-adventure-system-v1
git push --force-with-lease origin agent/woof-adventure-system-v1
```

Then run the full qualification pipeline again. A green run from the old stacked merge SHA is not release evidence for the rebased candidate.

## Go/no-go rule

Go only when all of the following are true:

- exact-head Adventure System CI is green;
- the parent stack has landed and the rebased candidate is requalified;
- production feature flag default is disabled until deliberate activation;
- reward and quest concurrency tests are green against PostgreSQL;
- Health emergency boundary remains free of game treatment;
- rollback requires no destructive database action.
