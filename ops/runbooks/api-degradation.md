# API degradation and caregiver authority incident runbook

Runbook ID: `api-degradation`  
Repository qualification: `CODE_QUALIFIED`  
Live rehearsal status: `NOT_YET_PROVEN`

## Authority boundary

This runbook covers application degradation after process and database readiness have been separated from user-facing correctness. The authoritative alert policy remains `ops/alerts/woof-api-alert-policy.v1.json`; low traffic that does not meet a ratio or latency sample floor is `INSUFFICIENT_DATA`, not healthy evidence.

Today/read latency authority currently covers exact controller operations:

- `AdventureController.getMine`;
- `CompanionController.getState`;
- `CompanionController.getReadiness`;
- `CaregiverController.getCaregiverToday`.

Caregiver transition authority covers exact issue, accept, decline, and revoke operations. Caregiver access remains capability-gated, replay-aware, block-aware, time-bounded, and context-only where observations are permitted. This runbook does not authorize weakening those semantics to recover availability.

## Detection

Use this runbook for:

- `todayReadP95Ms`;
- `caregiverTransition5xx`;
- `deviceContractRejections` when the problem presents as an API/contract regression rather than a specific connected provider;
- application `http5xx`;
- `application5xxBurst`;
- `requestDurationInvalid`.

Start by recording readiness and exact release identity:

```bash
export WOOF_API_ORIGIN="https://woof-api-prod.fly.dev"
curl --fail-with-body --silent --show-error \
  "$WOOF_API_ORIGIN/api/v1/ops/health/ready" | jq
```

When operational metrics authority is available:

```bash
test -n "${OPS_METRICS_TOKEN:-}" || { echo "OPS_METRICS_TOKEN unavailable" >&2; exit 2; }
curl --fail-with-body --silent --show-error \
  -H "x-woof-ops-token: ${OPS_METRICS_TOKEN}" \
  "$WOOF_API_ORIGIN/api/v1/ops/metrics.json" | jq
```

Do not print or persist the operational token.

## Impact

Classify the smallest failing authority surface before making changes:

- **Today/read latency regression:** one exact read operation is slow while readiness remains healthy;
- **caregiver transition availability:** issue/accept/decline/revoke returns server failures;
- **caregiver correctness:** transition succeeds or fails with the wrong authority semantics;
- **device envelope contract drift:** pre-provider envelopes are rejected before a trusted provider label exists;
- **instrumentation corruption:** request timing is invalid rather than genuinely slow;
- **broad API regression:** multiple unrelated operations fail on one release.

A 4xx authorization or validation outcome is not automatically an availability failure. Do not convert correct denials into success to reduce an error rate.

## Containment

1. Preserve the exact release SHA and operation name before changing state.
2. If only one optional product surface is implicated and it has a maintained feature/configuration boundary, prefer disabling that bounded surface over weakening shared auth, caregiver, validation, or reward authority.
3. Do not bypass `JwtAuthGuard`, caregiver capability checks, block checks, request-key replay protection, or observation context authority.
4. Do not manually edit caregiver grant rows to make a transition appear successful.
5. Do not discard invalid timing samples or rewrite them to zero. Broken timing is its own alert condition.
6. Do not infer that an endpoint is healthy solely because a ratio/latency alert is inactive below its sample floor.
7. If readiness is failing too, move first to `database-migration.md` or `deployment-readiness.md` rather than treating the symptom as endpoint-local.

## Diagnosis

For Today/read degradation, isolate one exact operation at a time. Use synthetic or non-private requests and compare:

- successful request count;
- p95 window state from external alert evaluation when available;
- 5xx count/ratio;
- invalid timing count;
- current release versus the immediately previous known-good release.

Do not treat Nest handler latency as browser, CDN, TLS, or full response-flush latency.

For caregiver transitions, exercise a synthetic grant lifecycle against maintained endpoints:

```text
POST /api/v1/caregiver/grants
POST /api/v1/caregiver/grants/:grantId/accept
POST /api/v1/caregiver/grants/:grantId/decline
POST /api/v1/caregiver/grants/:grantId/revoke
```

Use a unique synthetic request key for new issuance. Retrying the same issuance with the same semantics should be replay-safe; reusing its request key for different semantics must remain a conflict. Do not copy real grant IDs, user IDs, notes, or pet data into incident evidence.

For a device contract spike, distinguish:

- pre-provider envelope rejection, which increments `deviceContractRejections` without a provider label;
- a verified provider/kind import rejection, which belongs in `connector-ingestion.md`.

## Recovery

### Endpoint or release regression

Prefer a narrow forward fix when the defect is understood. If exact previous-image recovery is safer, use `deployment-readiness.md` and satisfy its database compatibility gate before redeploying an older image.

### Caregiver transition regression

Repair the transition implementation or persistence boundary while preserving:

- issuer/recipient authority;
- capability bundle constraints;
- block-state checks;
- expiry semantics;
- replay/idempotency behavior;
- context-only observation authority;
- no Bond XP or recommendation-evidence authority for caregiver observations.

Never recover by broadening caregiver permissions or restoring revoked/expired grants.

### Timing instrumentation regression

Repair the timing source or interceptor. Keep invalid samples excluded from latency histograms while continuing to count them as invalid. Do not suppress the alert merely because request outcomes are otherwise successful.

## Rollback

Application rollback follows `deployment-readiness.md`. A rollback must not:

- reverse or mutate caregiver state manually;
- restore a revoked caregiver grant;
- change reward eligibility to compensate for a broken flow;
- weaken authorization or validation;
- reinterpret insufficient sample data as a passed latency qualification.

If the failing release includes schema changes, follow `database-migration.md` before using an older image.

## Verification

Require the smallest representative synthetic journey plus alert recovery:

1. `/ops/health/ready` is healthy and reports the intended release SHA.
2. Each affected Today/read endpoint returns the expected bounded response under an authorized synthetic account.
3. A caregiver issuance can be replayed safely with the same request key and semantics.
4. Reusing that request key for different issuance semantics remains rejected.
5. Accept/decline/revoke preserve actor authority and terminal-state semantics.
6. A revoked, expired, blocked, or under-capability caregiver cannot regain access through the recovered path.
7. Caregiver observations remain `CONTEXT_ONLY` and do not become Bond XP or recommendation authority.
8. Invalid request-duration samples stop increasing after an instrumentation fix.
9. The triggering alert clears for its configured evaluation window, or low-volume evidence is explicitly labeled `INSUFFICIENT_DATA` rather than `OK`.

## Evidence

Capture only low-cardinality operational evidence:

- alert key and exact controller operation;
- release SHA;
- bounded status-class/count/latency classification;
- synthetic fixture label;
- recovery SHA/image;
- verification timestamps and pass/fail outcomes.

Do not retain real caregiver identities, pet IDs, grant IDs, request bodies, observation notes, device payloads, tokens, or query parameters.

## Rehearsal

Live rehearsal status: `NOT_YET_PROVEN`

Promotion to `PROVEN` requires a production-like synthetic drill that exercises at least one Today/read degradation and the complete caregiver issue/accept/revoke authority path, demonstrates containment without weakening authorization, and verifies recovery against an exact release.