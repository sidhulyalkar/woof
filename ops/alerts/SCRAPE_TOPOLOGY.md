# Operational Metrics Scrape Topology

Woof's API operational counters are intentionally **process-local**. They are not a shared global ledger.

This matters once the API has more than one replica.

## Required production scrape shape

A Prometheus-compatible collector must scrape each API process through a stable target identity, or receive equivalent instance-aware telemetry from the deployment platform.

Do **not** configure one scraper target that points at a load-balanced `/ops/metrics` URL when successive scrapes can land on different API processes. That makes independent counters appear to jump backward and forward inside one target series, which can corrupt `rate()`, `increase()`, and histogram calculations.

A correct multi-replica shape is:

1. each API process exposes its own process-local counters;
2. the collector gives each target a stable external identity such as `instance`;
3. `rate()` or `increase()` is evaluated on each underlying series before Woof's rules aggregate by bounded product labels such as `release`, `operation`, `provider`, or `kind`;
4. target health such as Prometheus `up` is retained outside Woof-owned application labels;
5. a rolling deployment may expose multiple `release` values concurrently, and they remain distinct until the old release disappears.

## Why the current rules aggregate by release

Phase 2 rules deliberately describe release-level product health and selected operation-level health. They do not claim per-replica diagnosis because the production scrape topology has not yet been selected and qualified.

When stable instance-aware scraping is configured, external target labels remain available for dashboards and provider-native target alerts even though Woof's alert expressions aggregate request evidence by release or operation.

## Target loss

`WoofOperationalMetricsMissing*` detects the absence of the Woof API metric family at the rule evaluator. `WoofReadinessProbeMissing*` checks that each visible release has readiness-request evidence.

Neither can identify a replica that the collector never sees while another replica remains visible under the same release. Production observability therefore also requires provider-native per-target scrape health, such as Prometheus `up`, plus a verified route to an operator.

## Counter resets

Process restarts reset process-local counters. Prometheus-compatible `rate()` and `increase()` calculations are designed to account for monotonic counter resets when each process retains a stable time-series identity. They cannot repair a scrape target that alternates between unrelated processes.

## Evidence boundary

Until the scrape topology and target-health routing are exercised against a deployed stack, the correct claim remains:

**Woof has code-qualified process-local metrics and alert rules, but multi-replica scrape identity and live target observability remain production evidence under #100.**
