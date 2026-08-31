# Woof API Alert Policy v1

This document defines the first code-qualified operational alert boundary for the Woof API.

It advances Phase 2 of #100. It does **not** claim that production scraping, alert routing, paging, or operator drills are live. Those remain deployment- and provider-dependent evidence.

## Authority model

There is one versioned source for initial thresholds:

- `ops/alerts/woof-api-alert-policy.v1.json`

Prometheus-compatible rules are generated from that policy:

- `ops/alerts/woof-api.rules.yml`

The generator and deterministic threshold fixtures live at:

- `.github/scripts/woof-alert-policy.py`
- `ops/alerts/fixtures/woof-api-alert-fixtures.v1.json`

CI regenerates the rule file and fails if committed rules drift from the policy. The generator also validates threshold ordering, minimum sample floors, exact controller-operation bindings, low-cardinality service identity, and required alert grouping. A ratio policy is rejected when its minimum sample floor would make warning and critical mathematically indistinguishable.

The rendered rules are independently parsed by an official pinned Prometheus `promtool` binary with a verified archive checksum. This protects against a generator producing text that looks plausible but is not valid Prometheus rule syntax.

These thresholds are **initial operator guardrails**, not an SLA, SLO, availability guarantee, or claim about production capacity. They should be tuned only after staging and live evidence exists.

## Exact operation authority

Alert targets are stored as exact `Controller.method` operations, not free-form regular expressions. The generator converts those exact names into RE2-safe Prometheus matchers and verifies that every referenced controller method still exists in source.

The Observability + Device workflow is triggered when those controller files change. Renaming a protected operation without updating alert policy therefore fails closed instead of silently turning off an alert.

Current source-bound controllers are:

- `AuthController`;
- `AdventureController`;
- `CompanionController`;
- `CaregiverController`;
- `ObservabilityController`.

## Metric identity and privacy

Operational metrics remain deliberately low-cardinality and identifier-free.

Every Woof-owned series carries:

- `service="woof-api"`;
- `release="<exact 40-hex Git SHA>"` or the explicit fallback `release="unknown"`.

Request series additionally use only bounded operational labels:

- HTTP method;
- controller/handler operation name;
- status class such as `2xx`, `4xx`, or `5xx`.

Alert rules add a static `operation_class` such as `today_read`, `auth_session`, `caregiver_transition`, or `telemetry_quality`.

The metrics and alert policy must not collect or label:

- user identifiers;
- pet identifiers;
- emails or handles;
- request URLs or route parameters;
- request bodies or query strings;
- provider external object identifiers;
- raw private payloads.

External infrastructure may add ordinary scrape labels such as `job` or `instance`. Woof rules aggregate process-local counters only across explicitly chosen bounded labels such as release, exact controller operation, or verified connector provider/kind.

## Request latency histogram

`woof_http_request_duration_ms` is a cumulative Prometheus histogram with fixed millisecond buckets:

`25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, +Inf`

This replaces percentile guessing from `sum`, `count`, or `max`. External Prometheus-compatible aggregation can now use `histogram_quantile()` over bounded windows.

The measured duration is **Nest request-handler execution time**, from interceptor entry until the controller observable emits or errors. It is not browser round-trip time, CDN latency, TLS time, or full response-flush latency. Those require external black-box or edge instrumentation.

The first latency alert targets successful `2xx` responses for each user-facing Today/read operation independently:

- `AdventureController.getMine`;
- `CompanionController.getState`;
- `CompanionController.getReadiness`;
- `CaregiverController.getCaregiverToday`.

Per operation, after at least 20 successful requests in 10 minutes:

- warning: p95 >= 750 ms for 10 minutes;
- critical: p95 >= 1500 ms for 5 minutes.

Per-operation grouping prevents a high-volume fast endpoint from statistically hiding a quieter slow endpoint.

These values are launch guardrails to surface regressions, not performance guarantees.

### Invalid timing samples

Non-finite or negative request durations are **not** coerced to `0 ms`. Recording bad timing as zero would bias the histogram toward looking faster when instrumentation is broken.

Instead:

- the request itself still increments `woof_http_requests_total`;
- the invalid duration is excluded from histogram buckets, sum, and count;
- `woof_http_request_duration_invalid_total` increments for that bounded operation/status series;
- warning begins at 1 invalid timing sample in 15 minutes;
- critical begins at 5 invalid timing samples in 15 minutes.

Connector duration aggregates use the same honest sampling rule and expose `woof_connector_import_duration_invalid_total`, although connector rejection alerts are based on import outcomes rather than timing.

## Initial alert classes

### Missing operational telemetry

The API process uptime metric is treated as the heartbeat for the operational scrape surface.

- warning when absent for 5 minutes;
- critical when absent for 15 minutes.

Missing telemetry is an **unknown/degraded observability state**, never evidence that the service is healthy.

This is a service-level metric heartbeat. Per-target scrape failure should later be covered by provider-native target health such as Prometheus `up`, because Woof cannot infer the identity of a target that is not being scraped at all.

### Unknown release identity

`woof_release_info{release="unknown"}` warns after 5 minutes.

A process without an exact Git SHA may run, but its evidence cannot be promoted as an exact qualified release.

### Readiness

Readiness is database-backed and intentionally separate from liveness.

The missing-probe rule is release-aware: every currently scraped release identity must have a corresponding readiness request series. A healthy old release therefore cannot hide the fact that a newly deployed release is never being probed.

The rules alert when:

- a scraped release has no readiness probe telemetry for 5/15 minutes;
- one readiness 5xx occurs in a 5-minute window and persists for the warning interval;
- three readiness 5xx responses occur in a 5-minute window for the critical boundary.

A process exporting metrics without an externally exercised readiness series is not treated as fully observed.

### Application HTTP 5xx ratio

The service-wide ratio intentionally excludes `ObservabilityController` liveness, readiness, metrics, and metrics-JSON operations. Probe/scrape traffic can otherwise dominate a low-traffic beta denominator and hide failures on real application paths.

The denominator also excludes `4xx` client outcomes. The operational server-error ratio is therefore `5xx / (2xx + 3xx + 5xx)`, not `5xx / all requests`. Invalid credentials, authorization denials, and malformed client traffic cannot make server reliability look artificially better.

Over a 5-minute window, after at least 50 eligible application requests:

- warning: 5xx ratio >= 2%;
- critical: 5xx ratio >= 5%.

At the minimum floor those severities remain discretely distinguishable; policy validation rejects configurations where the same number of failures would trigger both warning and critical.

### Authentication/session server errors

The auth/session class includes exact operations for:

- register;
- login;
- logout;
- logout-all;
- current-profile/session verification.

Each operation is evaluated independently. A high-volume healthy login path cannot hide a broken logout or session-verification operation.

Over 5 minutes, per operation, after at least 50 eligible `2xx + 3xx + 5xx` outcomes:

- warning: 5xx ratio >= 2%;
- critical: 5xx ratio >= 5%.

Expected authentication denials such as invalid-credential `401` responses are excluded from both numerator and denominator. They remain visible in the raw HTTP metrics but do not dilute the server-failure ratio.

### Caregiver authority transitions

The transition class includes exact issue, accept, decline, and revoke operations. Each operation is evaluated independently.

Over 5 minutes, per operation, after at least 25 eligible `2xx + 3xx + 5xx` outcomes:

- warning: 5xx ratio >= 2%;
- critical: 5xx ratio >= 5%.

Authorization/client `4xx` outcomes are not silently reclassified as server failures and do not dilute this server-error ratio. Correctness of allow/deny decisions remains covered by the dedicated caregiver authority contracts.

### Connector rejection ratio

Connector rejection is evaluated per exact verified `provider + kind` pair. A healthy provider cannot hide contract drift from another provider.

Over 15 minutes, after at least 20 verified import attempts for that provider/kind:

- warning: rejected ratio >= 10%;
- critical: rejected ratio >= 25%.

This surfaces partner/contract drift without recording provider payload contents or external identifiers.

### Device contract rejection count

Over 15 minutes:

- warning: at least 5 pre-provider contract rejections;
- critical: at least 20.

Operators should inspect transport/provider contract state without copying raw device payloads into tickets or chat.

## Insufficient data is not healthy evidence

Ratio and latency threshold fixtures have explicit minimum sample floors.

When telemetry is present but the minimum floor has not been met, deterministic policy evaluation reports `INSUFFICIENT_DATA`, not `OK`. Prometheus alert rules simply remain inactive until their sample-floor predicate is satisfied.

This distinction matters during low traffic and immediately after process startup.

The JSON fixtures validate deterministic threshold classification. They are **not** a substitute for Prometheus's own temporal `for:` state machine. `promtool check rules` independently validates the rendered rule syntax; live/staging alert drills remain required before claiming paging behavior.

## Warning versus critical routing

Warning and critical rules may both be logically true at critical severity. When Alertmanager or another external router is configured, the routing configuration should inhibit the warning notification when the matching critical alert for the same service/release/operation identity is firing. That routing behavior is not claimed until Phase 6 live operational qualification.

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

Run deterministic threshold fixtures:

```bash
python .github/scripts/woof-alert-policy.py --verify-fixtures
```

CI additionally:

- verifies alert target controller methods still exist;
- verifies threshold/sample-floor semantics;
- validates the rendered rule file with a checksum-pinned official Prometheus `promtool`;
- type-checks/tests the API metric implementation;
- verifies the alert and metric surfaces remain identifier-free.

## What remains before public-beta operational authority

This Phase 2 code release is not equivalent to a functioning pager.

Still required under #100 and #77:

1. connect a real external scraper/aggregator to the protected production metric endpoint;
2. configure and verify alert routing to an accountable operator, including warning inhibition beneath matching critical alerts;
3. exercise a benign alert drill and retain evidence tied to an exact deployed SHA;
4. add and rehearse incident runbooks for authentication, database, deployment, realtime authorization, and privacy failures;
5. perform a real backup/restore rehearsal against a production-equivalent snapshot;
6. run bounded load qualification using synthetic/non-production data;
7. add real realtime and database-pool signals before claiming those alert classes are covered;
8. add external black-box/edge latency evidence before making end-user latency claims.

Until those are completed, the correct statement is: **alert policy and metric contracts are code-qualified, while live operational routing and recovery evidence remain unproven.**
