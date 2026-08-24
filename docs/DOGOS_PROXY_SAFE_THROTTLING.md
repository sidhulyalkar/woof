# dogOS Proxy-Safe Throttling v1

This release hardens the existing global API throttler for production behind Fly Proxy without changing throttle limits, database schema, product authority, or user-facing feature semantics.

## Problem

Nest's default throttler tracks HTTP clients using the request IP. Behind a reverse proxy, the direct peer address can represent the proxy rather than the original client. If multiple public users collapse onto the same proxy address, they can unintentionally share one throttle bucket.

Woof deploys its API behind Fly Proxy. Fly documents `Fly-Client-IP` as the client address observed by Fly Proxy and recommends it over manually parsing `X-Forwarded-For` when Fly is the direct edge. Fly also documents `FLY_APP_NAME` and `FLY_MACHINE_ID` as runtime environment values supplied to Machines.

## Trust boundary

Woof does not enable broad Express `trust proxy=true` and does not parse `X-Forwarded-For` for throttling.

The throttle tracker trusts `Fly-Client-IP` only when both Fly runtime identity variables are present:

- `FLY_APP_NAME`
- `FLY_MACHINE_ID`

The header must be exactly one valid IPv4 or IPv6 address. Arrays, comma-separated chains, empty values, and malformed addresses are rejected.

When the runtime is not demonstrably Fly-hosted, `Fly-Client-IP` is ignored even if a caller supplies it.

When the runtime is Fly-hosted but the Fly client header is absent or invalid, the tracker falls back to the direct request/socket address. This can conservatively merge clients into a proxy bucket, but it cannot create arbitrary attacker-controlled identities or bypass rate limiting.

## Tracker identities

Tracker keys are source-prefixed so trust domains cannot collide accidentally:

- `fly:<validated-client-ip>` for a validated Fly client address in a Fly runtime;
- `fly-fallback:<direct-peer-ip>` when Fly runtime is proven but the client header is unusable;
- `direct:<direct-peer-ip>` everywhere else.

The value is used only as the internal throttler tracker. This release does not add client IPs to telemetry or application logs.

## Existing policy remains unchanged

The global limits remain:

- short: 3 requests per 1 second;
- medium: 20 requests per 10 seconds;
- long: 100 requests per 60 seconds.

Test environments continue to skip throttling under the existing policy. Production behavior keeps the global `ThrottlerGuard` and changes only its tracker source.

## Qualification

One exact candidate head must pass every workflow triggered by this ownership surface.

The dedicated Proxy-Safe Throttling lane must remain schema-free and prove:

1. all inherited migrations still apply to PostgreSQL;
2. formatting, zero-warning lint, API TypeScript, focused unit tests, and API build pass;
3. `throttlerOptions` uses the narrow `clientIpTracker`;
4. production source does not enable broad `trust proxy` or use `X-Forwarded-For` for throttle identity;
5. a spoofed Fly client header is ignored outside a complete Fly runtime identity;
6. malformed or multi-value Fly client headers fail safe to the direct peer;
7. valid IPv4 and IPv6 Fly client addresses are accepted only inside a proven Fly runtime;
8. the real production Docker image boots against PostgreSQL with simulated Fly runtime identity;
9. three rapid requests from Fly client A succeed and the fourth is throttled with HTTP 429;
10. a request from distinct Fly client B immediately succeeds, proving A and B do not share the short bucket.

Diagnostic heads do not qualify a release.
