# Woof API Alert Policy v1

This document defines the first code-qualified operational alert boundary for the Woof API.

It advances Phase 2 of #100. It does **not** claim that production scraping, alert routing, paging, or operator drills are live. Those remain deployment- and provider-dependent evidence.

## Authority model

There is one versioned source for initial thresholds:

- `ops/alerts/woof-api-alert-policy.v1.json`

Prometheus-compatible rules are generated from that policy:

- `ops/alerts/woof-api.rules.yml`

The generator and deterministic fixture evaluator live at:

- `.github/scripts/woof-alert-policy.py`
- `ops/alerts/fixtures/woof-api-alert-fixtures.v1.json`

CI regenerates the rule file and fails if committed rules drift from the policy.

These thresholds are **initial operator guardrails**, not an SLA, SLO, availability guarantee, or claim about production capacity. They should be tuned only after staging and live evidence exists.

## Metric identity and privacy

Operational metrics remain deliberately low-cardinality and identifier-free.

Every Woof-owned series carries:

- `service="woof-api"`
- `release="<exact 40-hex Git SHA>"` or the explicit fallback `release="unknown"`

Request series additionally use only bounded operational labels:

- HTTP method
- controller/handler operation name
- status class such as `2xx`, `4xx`, or `5xx`

Alert rules add a static `operation_class` such as `today_read`, `auth_session`, or `caregiver_transition`.

The metrics and alert policy must not collect or label:

- user identifiers;
- pet identifiers;
- emails or handles;
- request URLs or route parameters;
- request bodies or query strings;
- provider external object identifiers;
- raw private payloads.

External infrastructure may add ordinary scrape labels such as `job` or `instance`. Woof's rules intentionally aggregate process-local counters by `release` so multiple replicas of one release can be evaluated together.

## Request latency histogram

`woof_http_request_duration_ms` is a cumulative Prometheus histogram with fixed millisecond buckets:

`25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, +Inf`

This replaces percentile guessing from `sum`, `count`, or `max`. External Prometheus-compatible aggregation can now use `histogram_quantile()` over bounded windows.

The first latency alert targets the user-facing Today/read composition path:

- `AdventureController.getMine`
- `CompanionController.getState`
- `CompanionController.getReadiness`
- `CaregiverController.getCaregiverToday`

Initial p95 guardrails after at least 20 requests in 10 minutes:

- warning: p95 >= 750 ms for 10 minutes;
- critical: p95 >= 1500 ms for 5 minutes.

These values are launch guardrails to surface regressions, not performance guarantees.

## Initial alert classes

### Missing operational telemetry

The API process uptime metric is treated as the heartbeat for the operational scrape surface.

- warning when absent for 5 minutes;
- critical when absent for 15 minutes.

Missing telemetry is an **unknown/degraded observability state**, never evidence that the service is healthy.

### Unknown release identity

`woof_release_info{release="unknown"}` warns after 5 minutes.

A process without an exact Git SHA may run, but its evidence cannot be promoted as an exact qualified release.

### Readiness

Readiness is database-backed and intentionally separate from liveness.

The rules alert when:

- readiness probe telemetry itself is missing for 5/15 minutes;
- one readiness 5xx occurs in a 5-minute window and persists for the warning interval;
- three readiness 5xx responses occur in a 5-minute window for the critical boundary.

A process exporting metrics without an externally exercised readiness series is not treated as fully observed.

### Service-wide HTTP 5xx ratio

Over a 5-minute window, after at least 20 requests:

- warning: 5xx ratio >= 2%;
- critical: 5xx ratio >= 5%.

The denominator includes all measured HTTP operations for the release.

### Authentication/session server errors

The auth/session class includes:

- register;
- login;
- logout;
- logout-all;
- current-profile/session verification.

Over 5 minutes, after at least 10 requests:

- warning: 5xx ratio >= 2%;
- critical: 5xx ratio >= 5%.

Expected authentication denials such as an invalid-credential `401` are not counted as server failures because this alert is explicitly based on `status_class="5xx"`.

### Caregiver authority transitions

The transition class includes issue, accept, decline, and revoke operations.

Over 5 minutes, after at least 10 transitions:

- warning: 5xx ratio >= 2%;
- critical: 5xx ratio >= 5%.

Authorization denials and ordinary client errors are not silently reclassified as server failures.

### Connector rejection ratio

Over 15 minutes, after at least 20 verified import attempts:

- warning: rejected ratio >= 10%;
- critical: rejected ratio >= 25%.

This is intended to surface partner/contract drift without recording provider payload contents.

### Device contract rejection count

Over 15 minutes:

- warning: at least 5 pre-provider contract rejections;
- critical: at least 20.

Operators should inspect transport/provider contract state without copying raw device payloads into tickets or chat.

## Insufficient data is not healthy evidence

Ratio and latency fixtures have explicit minimum sample floors.

When telemetry is present but the minimum floor has not been met, deterministic policy evaluation reports `INSUFFICIENT_DATA`, not `OK`. Prometheus alert rules simply remain inactive until their sample-floor predicate is satisfied.

This distinction matters during low traffic and immediately after process startup.

## Explicitly deferred signals

Two #100 alert classes are intentionally **not fabricated** in v1:

### Realtime admission/error rate

There is not yet a dedicated low-cardinality realtime admission/error series. HTTP counters are not a valid proxy for stale socket authority, admission failures, disconnects, or revocation propagation.

### Database connection/pool saturation

Database-backed readiness proves a query can succeed. It is not a connection-pool saturation gauge. No pool-capacity alert will be claimed until a privacy-safe source exists.

The policy JSON records both deferred signals and their reasons so omission cannot masquerade as completion.

## Regeneration and verification

Render rules:

```bash
python .github/scripts/woof-alert-policy.py > ops/alerts/woof-api.rules.yml
```

Verify the committed file matches policy:

```bash
python .github/scripts/woof-alert-policy.py --check
```

Run deterministic policy fixtures:

```bash
python .github/scripts/woof-alert-policy.py --verify-fixtures
```

CI also type-checks/tests the API metric implementation and verifies the alert surface remains identifier-free.

## What remains before public-beta operational authority

This Phase 2 code release is not equivalent to a functioning pager.

Still required under #100 and #77:

1. connect a real external scraper/aggregator to the protected production metric endpoint;
2. configure and verify alert routing to an accountable operator;
3. exercise a benign alert drill and retain evidence tied to an exact deployed SHA;
4. add and rehearse incident runbooks for authentication, database, deployment, realtime authorization, and privacy failures;
5. perform a real backup/restore rehearsal against a production-equivalent snapshot;
6. run bounded load qualification using synthetic/non-production data;
7. add real realtime and database-pool signals before claiming those alert classes are covered.

Until those are completed, the correct statement is: **alert policy and metric contracts are code-qualified, while live operational routing and recovery evidence remain unproven.**
