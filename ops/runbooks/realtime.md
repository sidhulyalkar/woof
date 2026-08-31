# Realtime authorization incident runbook

Runbook ID: `realtime`  
Repository qualification: `CODE_QUALIFIED`  
Live rehearsal status: `NOT_YET_PROVEN`

## Authority boundary

Woof realtime chat is server-authoritative. A socket is admitted only after JWT verification yields a user ID, session ID, and valid expiry, and the database-backed session authority confirms that session is active. Sensitive socket actions re-check active-session authority before work is accepted.

Current maintained realtime semantics include:

- session expiry clears socket authority, emits `session:expired`, and disconnects;
- revoked/inactive session detection clears authority, emits `session:revoked`, and disconnects;
- chat messages are persisted before successful delivery;
- conversation access and recipient eligibility are checked through chat security;
- recipients are filtered against currently active sessions before emission;
- message, typing, and membership actions have bounded in-memory admission policies;
- arbitrary exception messages are not logged; realtime rejection telemetry currently exposes only bounded exception class names.

There is **no dedicated low-cardinality realtime admission/error metric yet**. `realtime_admission_error_rate` remains manual/deferred. HTTP request counters are not a substitute for socket health.

## Detection

Use this runbook for manual/deferred signal `realtime_admission_error_rate`, or reports of:

- valid sessions unable to establish realtime state;
- revoked or expired sessions continuing to act;
- unauthorized users receiving conversation events;
- block/conversation membership changes not taking effect;
- unexpected message duplication or delivery before persistence;
- admission/rate-limit storms;
- sockets remaining connected after authority is revoked.

Because no maintained realtime health metric exists, detection currently requires an authorized synthetic client journey, bounded server logs, or direct operator observation. Label evidence `MANUAL_DEFERRED` rather than pretending it came from Prometheus.

## Impact

Classify:

- **connection admission:** valid synthetic session cannot reach `session:ready`;
- **stale session authority:** revoked/expired session remains usable;
- **recipient authorization:** a user outside the authorized recipient set receives message/typing events;
- **conversation authorization:** join/send/typing succeeds without current conversation access;
- **delivery integrity:** an unpersisted or duplicate message is emitted as new;
- **rate-limit abuse:** admission buckets reject excessive message/typing/membership traffic;
- **general runtime failure:** realtime is affected because database/auth/application runtime is unhealthy.

Treat stale-session or unauthorized-recipient behavior as a security incident even when ordinary HTTP/readiness is green.

## Containment

1. Do not disable session revalidation, chat-security recipient filtering, conversation access checks, or realtime admission limits to restore connectivity.
2. If the issue is session compromise, use `auth-session.md` for user/global revocation authority.
3. If unauthorized recipient delivery is confirmed and there is no dedicated realtime kill switch, stop or roll forward/back the affected API release through the production runtime authority. Do not invent an undocumented socket bypass or operator endpoint.
4. Preserve bounded release and event-class evidence without copying chat text, room IDs tied to users, JWTs, session IDs, or private conversation payloads.
5. If the failure follows database/readiness degradation, follow `database-migration.md` before treating it as a socket-local defect.
6. Do not increase rate limits during an incident merely to make rejection symptoms disappear. Determine whether the traffic is legitimate, abusive, or a client retry loop first.

## Diagnosis

Start with HTTP liveness/readiness to exclude broad process/database failure:

```bash
export WOOF_API_ORIGIN="https://woof-api-prod.fly.dev"
curl --fail-with-body --silent --show-error \
  "$WOOF_API_ORIGIN/api/v1/ops/health/live" | jq
curl --fail-with-body --silent --show-error \
  "$WOOF_API_ORIGIN/api/v1/ops/health/ready" | jq
```

Then use an authorized synthetic account pair and inspect only bounded outcomes:

1. connect with a valid JWT/session and require `session:ready`;
2. connect with missing/invalid/expired auth and require disconnect without ready state;
3. revoke the synthetic session through maintained auth authority, then attempt a sensitive realtime action;
4. verify the socket is rejected/disconnected and cannot continue sending;
5. exercise a conversation available to both synthetic users and a conversation unavailable to one user;
6. verify unauthorized join/send/typing is rejected;
7. verify only authorized active recipients receive the event;
8. exercise admission thresholds using synthetic traffic and confirm bounded `rate_limited` responses.

Current rate policies are process-local:

- message: 5 per second and 60 per minute;
- typing: 8 per 5 seconds and 60 per minute;
- membership: 10 per 5 seconds and 60 per minute.

Do not treat these process-local buckets as a distributed/global abuse limit across replicas.

## Recovery

### Session-authority regression

Repair the admission/revalidation path while preserving database-backed session checks on connect and sensitive actions. If a session is revoked or expired, recovery must continue to remove its socket authority rather than merely hiding the disconnect signal.

### Recipient/conversation authorization regression

Repair chat-security and active-session filtering. Prefer fail-closed delivery over broadcasting to an uncertain recipient set.

### Delivery-integrity regression

Restore persist-before-emit behavior and idempotent client-message handling. Do not emit a successful message before canonical persistence succeeds.

### Rate-limit/client retry regression

Fix the client/server retry behavior while preserving bounded admission. Any policy change should be a reviewed product/abuse-control decision, not incident-time convenience.

### Release regression

Use a reviewed forward fix or `deployment-readiness.md` exact-image recovery subject to database compatibility.

## Rollback

Realtime code rollback follows `deployment-readiness.md` and must preserve:

- database-backed active session authority;
- expiry/revocation disconnect semantics;
- conversation membership/block authorization;
- active-session recipient filtering;
- persist-before-emit messaging;
- idempotent message semantics;
- bounded admission policies.

Do not roll back to a release that relied on token verification alone without server-side session authority or that broadcast realtime presence/events globally.

## Verification

Using synthetic accounts and conversations, require:

1. valid active session reaches `session:ready`;
2. invalid/expired session cannot become ready;
3. current-session or all-session revocation prevents subsequent sensitive socket actions and disconnects the affected socket;
4. token expiry emits `session:expired` and removes authority;
5. revoked authority emits `session:revoked` when detected and removes authority;
6. unauthorized conversation join/send/typing remains rejected;
7. blocked/ineligible/revoked recipients do not receive new events;
8. one logical message is persisted once and emitted as new at most once;
9. message/typing/membership admission policies still return bounded rate-limit outcomes under synthetic excess traffic;
10. HTTP readiness remains healthy on the intended release;
11. realtime evidence is still labeled manual/deferred until a dedicated privacy-safe metric exists.

## Evidence

Capture only:

- exact release SHA;
- synthetic fixture label;
- event class (`connect`, `message`, `typing`, `membership`, `revocation`, `expiry`);
- bounded outcome (`ready`, `unauthorized`, `rate_limited`, `revoked`, `expired`, `delivered`);
- timestamps;
- recovery SHA/image and verification pass/fail.

Never capture JWTs, session IDs, user IDs, conversation IDs from real users, chat text, media URLs, socket handshake auth, block targets, or raw exception payloads.

## Rehearsal

Live rehearsal status: `NOT_YET_PROVEN`

Promotion to `PROVEN` requires a production-like synthetic socket drill that demonstrates valid admission, session revocation/expiry propagation, unauthorized-conversation rejection, authorized-recipient filtering, persist-before-emit/idempotency behavior, and bounded admission under excess traffic. A dedicated realtime metric remains a separate prerequisite before realtime alert routing can be called code-qualified.