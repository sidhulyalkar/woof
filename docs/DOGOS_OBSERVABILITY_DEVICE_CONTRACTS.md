# dogOS Observability + Device Partner Contracts v1

## Purpose

This release makes dogOS operationally measurable and partner-integration-ready without creating a second source of pet truth or expanding location collection.

It has three jobs:

1. report real service liveness/readiness rather than optimistic placeholders;
2. expose low-cardinality operational metrics that do not contain user, pet, provider-object, payload, or request-URL identifiers;
3. define a bounded, versioned wearable transport envelope that enters the already-qualified Connectors → Autopilot → CareEvent import path.

## Health semantics

`GET /api/v1/ops/health/live` reports process liveness only. It does not query Postgres.

`GET /api/v1/ops/health/ready` performs a real `SELECT 1` through the canonical Prisma client. A database failure returns HTTP 503 through `ServiceUnavailableException`.

The legacy `GET /api/v1/health` surface now uses the same real readiness probe and reports `healthy` or `degraded`; it no longer hard-codes a connected database.

## Operational metrics

`GET /api/v1/ops/metrics` exposes Prometheus text and `GET /api/v1/ops/metrics.json` exposes the same process-local aggregates as JSON.

Both endpoints fail closed unless `OPS_METRICS_TOKEN` is configured and the caller supplies the exact value in `x-woof-ops-token`. Token comparison hashes both values to fixed-length SHA-256 digests before `timingSafeEqual`. A configured token must contain at least 32 characters.

Metrics are deliberately process-local. Horizontal deployments should scrape and aggregate externally rather than persisting per-request telemetry in the application database.

### Allowed labels

HTTP metrics may contain only:

- HTTP method;
- Nest controller/handler operation name;
- HTTP status class (`2xx`, `4xx`, `5xx`, etc.);
- aggregate count, total duration, and max duration.

Verified device-import metrics may contain only:

- provider from the closed connector registry;
- event kind from the closed wearable event enum;
- outcome (`IMPORTED`, `DUPLICATE`, `REJECTED`);
- aggregate count and timing.

Malformed envelopes rejected before a trusted provider/kind exists increment one unlabeled counter: `woof_device_contract_rejections_total`.

Operational metrics must not contain:

- user IDs;
- pet IDs;
- provider account/pet/object IDs;
- raw provider payloads;
- request URLs, query strings, or route parameters;
- precise location data.

Product analytics remains a separate subsystem. Operational metrics are not behavioral analytics and must not be repurposed as one.

## Device partner envelope

The internal transport contract is `woof-device-partner-v1`.

Required fields:

- `schemaVersion` exactly `woof-device-partner-v1`;
- wearable `provider` (`FI` or `TRACTIVE` in v1);
- bounded `externalPetId` and `externalObjectId` identifiers;
- `kind` of `DAILY_ACTIVITY` or `DEVICE_STATUS`;
- valid `observedAt` timestamp;
- JSON-object `payload`.

V1 limits:

- maximum payload: 16 KiB after JSON serialization;
- maximum future clock skew: 5 minutes;
- maximum historical backfill: 35 days;
- identifiers: non-empty, no control characters, at most 160 characters;
- recursive precise-location keys are rejected before provider normalization.

The contract is an **internal verified-transport seam**. No public browser controller accepts provider-originated wearable events.

## Authority and source truth

The partner envelope does not gain new product authority.

After validation, it is reduced to the existing `VerifiedWearableTransportEvent` and sent through `ConnectorsService.ingestWearableFromTransport()`. That path still requires:

1. a connected provider account;
2. a usable encrypted connector credential;
3. a provider pet mapped to a dogOS pet owned by the user;
4. deterministic provider normalization with precise-location rejection;
5. idempotency/hash replay checks;
6. delegation to Autopilot/CareEvent canonical source truth;
7. hash-only connector import provenance.

Device partner v1 does **not** permit:

- direct canonical Pet mutation;
- raw provider payload persistence;
- precise GPS/location import;
- browser provider impersonation;
- autonomous purchases;
- wearable-derived reward farming.

## Scaling posture

The first operational metrics implementation is intentionally in-memory and bounded by closed label sets. It is suitable for normal Prometheus-style scraping and avoids adding a telemetry write on every API request.

A later scaling release may add distributed tracing/export aggregation, queue-depth/worker metrics, connection-pool saturation, or provider-specific verified webhook transports. Those additions must preserve the same identifier and authority boundaries.

## Qualification

This release requires one exact head to pass:

1. `dogOS Observability + Device Contracts CI`;
2. root `CI`;
3. `dogOS Connectors CI`, because the release changes the qualified Connectors service and configuration surface.

Any other inherited dogOS workflow that is triggered by a changed owned path also becomes required for that exact head. Diagnostic heads do not count.
