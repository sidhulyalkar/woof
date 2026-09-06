# Bounded operational load qualification v1

Issue: #100, Phase 5

## Purpose

This lane answers a narrow pre-launch question: can one resource-bounded Woof production image preserve its maintained authorization, idempotency, health, and abuse-control contracts under a representative synthetic workload?

It is **not live-production proof**. It does not use production user data, infer production capacity, prove multi-machine scaling, prove database-provider headroom, or change `productionQualified` to true.

## Environment class

The committed `ci` and `prelaunch` profiles run against a fresh PostgreSQL database and the real production API image built from the exact qualified Git commit. On pull requests, the load workflow explicitly checks out the **literal PR head SHA** rather than GitHub's synthetic merge commit; workflow dispatch uses its exact dispatch SHA. Root CI remains responsible for merge-context compatibility, while this lane binds retained load evidence to the literal code/config revision under qualification.

The image boots with `NODE_ENV=production`, Adventure explicitly enabled, Swagger disabled, and the normal global throttler active. The workflow constrains the API container to:

- 1 GiB memory;
- 2 CPUs;
- 256 PIDs.

Those values are a qualification budget, not a claim that production needs exactly those resources.

The image receives the exact qualified SHA through `WOOF_RELEASE_SHA`. The harness requires both readiness and the protected operational-metrics snapshot to report that exact release before the run can qualify.

## synthetic-only fixture authority

The harness does not write fixture rows directly and never imports Prisma or the database package. It creates ephemeral synthetic state through maintained product APIs:

1. register an owner and caregiver through `/auth/register`, receiving real persisted sessions;
2. create an owned dog through `/pets`, which creates/repairs the deterministic personal household through normal runtime authority;
3. read the owner household and set its timezone to `UTC` through the maintained household API;
4. issue and accept a bounded caregiver grant;
5. capture Daily Signals through the maintained Intelligence API.

The CI database is disposable. No production account, pet, household, grant, credential, or event is read or copied.

The harness implementation is intentionally split into a thin orchestrator plus transport/evidence support, authority scenarios, and operational telemetry evaluation. Concurrency helpers receive their operation accumulator explicitly; transition evidence cannot depend on an undeclared property attached to a synthetic client object.

## Representative load profile

The `ci` profile uses seven owner/caregiver worker pairs for 24 seconds. The `prelaunch` profile uses fourteen pairs for 45 seconds. Each logical worker starts at most one representative request every 700 ms.

The measured rotation covers:

- persisted session verification with `AuthController.getProfile`;
- Adventure Today composition with `AdventureController.getMine`;
- Companion state with `CompanionController.getState`;
- Companion readiness with `CompanionController.getReadiness`;
- authorized caregiver Today with `CaregiverController.getCaregiverToday`;
- database-backed readiness with `ObservabilityController.readiness`.

Seven authenticated Socket.IO sessions are also admitted and held during the CI HTTP workload, then disconnected cleanly. This is a bounded connect/session-ready/disconnect proof. It is not a substitute for a dedicated realtime admission-error metric, which remains explicitly deferred in the alert policy.

## rate limiting stays active

The workflow does not switch to `NODE_ENV=test`, patch the throttler, or bypass the abuse controls.

A local production image would otherwise collapse every caller onto the same loopback throttle identity. To model the trusted Fly proxy boundary without disabling throttling, the container receives synthetic `FLY_APP_NAME` and `FLY_MACHINE_ID` values and each worker sends a distinct RFC 5737 documentation-only `fly-client-ip` address.

The representative 700 ms cadence remains below all current per-client HTTP limits. A separate six-request burst from one isolated synthetic client must produce both successful responses and at least one HTTP 429. The run fails if representative traffic receives a 429 or if the abuse burst does not.

## Retry and concurrency invariants

### Daily Signals

For each worker, four exact Daily Signals attempts are issued as two two-request concurrency waves after the short throttle window resets.

Qualification requires:

- one canonical `careEventId` across every response;
- exactly one primary receipt with `duplicate: false`;
- all retries to resolve through canonical duplicate semantics;
- a same-day divergent payload to fail closed with HTTP 409;
- Adventure `bondXp` to be identical before and after the complete load run.

This proves retry/concurrency does not mint a second canonical event or a hidden reward.

### Caregiver transitions

Issue, accept, revoke, and decline use the same two-wide repeated concurrency pattern. The run requires stable grant identity, replay evidence, and the correct terminal/effective state. With seven CI workers, each accept/revoke/decline operation receives 28 successful transition requests, exceeding the canonical alert policy's 25-request sample floor. Issue receives both active and decline-fixture waves.

Caregiver Today must work while the active grant exists and fail after revocation.

## operational metrics, not proxy latency theater

The harness retains its own localhost HTTP wall-clock p50/p95/p99/max as secondary evidence. Promotion decisions do not pretend that number is identical to production handler telemetry.

After load, the harness queries the protected low-cardinality `/ops/metrics.json` snapshot. That is the same privacy-safe operational metric source used by the alert policy. The snapshot must report:

- exact release identity;
- no user, pet, external-provider, raw-payload, or request-URL collection;
- zero 5xx responses across the run;
- zero invalid duration samples;
- at least the committed sample floors for auth, readiness, Today/read, and caregiver transitions.

For the four Today/read operations, the harness consumes the canonical `woof-api-alert-policy-v1` values directly. The committed minimum is 20 samples, warning boundary is **750 ms**, and critical boundary is **1500 ms**. Histogram p95 is retained as the first cumulative bucket containing the 95th-percentile sample. A critical result fails qualification. A warning result is retained explicitly rather than relabeled as green.

## Retained evidence and privacy

The uploaded `report.json` is deliberately sparse. It may contain only:

- exact expected and observed release SHA;
- environment/profile/resource-limit metadata;
- operation-class request counts and latency quantiles;
- low-cardinality operational metric summaries;
- bounded warning/failure codes;
- boolean invariants;
- aggregate rate-limit and realtime session-ready evidence.

It must not contain user, pet, household, grant, CareEvent, ledger, or socket identifiers; emails; passwords; bearer tokens; raw request/response bodies; free-form notes; request URLs; provider payloads; or ciphertext. The CI validator rejects those classes before artifact retention.

Evidence is retained for seven days through the repository-qualified `actions/upload-artifact@v7` runtime.

## What Phase 5 proves

A green literal-head run proves, for the named Git SHA and the `github-actions-production-image` environment class:

- the production image boots under explicit resource limits;
- real persisted sessions remain authorized under representative concurrency;
- Today/read and readiness operations meet sample floors without 5xx or invalid timing;
- Daily Signals retry identity remains canonical and reward-neutral;
- caregiver transition replay authority survives concurrency;
- bounded Socket.IO sessions can be admitted and disconnected during HTTP load;
- HTTP rate limiting is still active and representative traffic does not depend on disabling it;
- retained evidence remains privacy-safe and release-bound.

## What remains outside this proof

This lane is **not live-production proof** and does not close #100 by itself. Phase 4 backup/restore rehearsal and Phase 6 live operational qualification still require real deployment and provider authority. Production database pool saturation and a dedicated realtime admission/error signal also remain intentionally unproven rather than inferred from readiness or this CI workload.
