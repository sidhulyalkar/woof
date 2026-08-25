# dogOS Session Authority v1

Session Authority v1 turns Woof access tokens from self-contained bearer credentials into finite leases backed by server-owned session state. It builds directly on Realtime Session Lifetime v1 and does not replace JWT signature or expiry validation.

## Problem closed

Before this release, a valid signed access token remained authoritative until its JWT expiry. Browser and mobile logout only deleted the local token. The server had no persisted login-session identity, so it could not distinguish two devices, revoke one stolen token, revoke all sessions after a credential event, or reject an otherwise valid JWT after logout.

Realtime Session Lifetime v1 correctly enforced `exp`, but expiry alone is not revocation.

## Canonical authority model

Every successful register or login now creates a random server-owned session ID and places it in the access-token `sid` claim.

The server persists only the minimum session authority required for enforcement:

- session ID;
- user ID;
- expiry time;
- optional revocation time and bounded reason;
- creation time.

The session table deliberately stores no raw JWT, IP address, device fingerprint, browser user agent, location, refresh token, or other behavioral metadata.

An authenticated request is valid only when both layers pass:

1. JWT signature and finite expiry validation; and
2. `dogos_auth.sessions` contains the matching `sid` + `user_id` with no revocation and an expiry still in the future.

A token signed correctly but lacking `sid` fails closed.

## Login and registration

`AuthService` signs a token with `sub`, `email`, `handle`, and a fresh random `sid`, decodes the signed token only to obtain its canonical `exp`, and persists the session with that exact expiry before returning the token to the client.

If the token has no finite future expiry or session persistence fails, the login does not return an access token.

## HTTP revocation

Two authenticated endpoints become authoritative:

- `POST /api/v1/auth/logout` revokes the current `sid` for the current user.
- `POST /api/v1/auth/logout-all` revokes every still-active session for the current user.

`JwtStrategy` checks persisted session authority on every protected HTTP request. A revoked token therefore receives `401` even when its signature and `exp` remain valid.

`logout-all` first locks the user's active session rows in deterministic session-ID order, then revokes that exact set. Revocation is user-scoped, so one user's operation cannot alter another user's sessions.

## Realtime ingress

Socket.IO authentication now requires `sub`, finite `exp`, and `sid`. The persisted session must be active before the socket may enter its private `user:<userId>` room.

Inbound realtime actions preserve the established anti-abuse ordering:

1. resolve authenticated socket identity;
2. require the in-memory finite JWT lease;
3. validate and normalize the payload;
4. consume the authenticated-user admission budget;
5. acquire lock-bound persisted session authority;
6. perform canonical relationship authorization, persistence, or room work while that authority remains held.

The database-backed session boundary intentionally remains after rate admission. A client cannot create an unlimited session-table read workload by flooding malformed or already-rate-limited events.

`withActiveSession()` selects the actor's still-active session row with `FOR SHARE` and executes the admitted realtime action before releasing that lock. Multiple active operations for the same still-valid session may hold compatible `FOR SHARE` authority concurrently. Revocation updates conflict with those locks. Therefore, if revocation commits first the action never starts; if one or more actions acquire authority first, those already-admitted actions complete before revocation can commit. A stale pre-revocation read cannot authorize post-revocation persistence.

If persisted session authority is unavailable, the gateway emits `session:revoked { reason: 'session_revoked' }`, removes local socket authority, and server-disconnects that socket.

### Transaction ownership and pool safety

Lock-bound authority and canonical realtime database work share **one Prisma transaction client**. `withActiveSession()` passes its `Prisma.TransactionClient` into the admitted callback, and message persistence, conversation membership authorization, typing relationship authorization, and typing recipient-session filtering reuse that exact client.

This is a correctness and availability requirement, not merely an optimization. A lock-bound callback must never call a standalone service path that opens another Prisma transaction or uses the global Prisma client for canonical work. Otherwise several concurrent actions can occupy every pool connection with session-authority transactions and then deadlock while each waits for a second connection.

The transaction-bound message path therefore:

1. validates and normalizes the message before database work;
2. takes a transaction-scoped advisory lock on the caller-owned receipt identity;
3. rechecks relationship authorization under the same transaction;
4. resolves or creates the canonical message and receipt under that same transaction; and
5. returns the committed message result to the gateway.

Socket.IO message fanout intentionally happens **after** canonical persistence commits. Delivery then performs a fresh, short relationship/session-recipient authorization transaction. A transient fanout failure cannot roll back a valid persisted message, and the actor's session lock is not held across network emission.

Typing is different because it is ephemeral. Its relationship authorization and recipient-session filtering remain inside the actor's authority transaction, and only an already-authorized recipient set is emitted.

CI includes both a production concurrent flood/reconnect probe and a real PostgreSQL one-connection-pool regression. The latter sets `connection_limit=1`; valid lock-bound message work must still complete because it never asks Prisma for a second connection while the authority transaction is open.

## Transport connectivity is not authorization readiness

Socket.IO's client-side `connect` event proves that the transport handshake completed. It does **not** prove that Nest's asynchronous persisted-session admission has finished. Treating those two moments as equivalent creates a race where a valid user can emit immediately after transport connection while the gateway has not yet populated its authoritative socket/session maps.

Session Authority therefore defines an explicit application-level readiness boundary:

1. Socket.IO transport connects;
2. the server verifies JWT signature, finite `exp`, `sub`, and `sid`;
3. `SessionAuthorityService.withActiveSession()` proves the persisted session is active;
4. the gateway installs the socket's in-memory user, session, and finite-expiry authority;
5. the socket joins its private `user:<userId>` room;
6. the server rechecks that the transport did not disappear during asynchronous admission; and
7. only then does the server emit `session:ready { socketId: client.id }`.

The `socketId` is an authorization-epoch binding, not a user identifier. The web client accepts readiness only when the payload matches the current Socket.IO `socket.id`. A delayed `session:ready` from a previous transport epoch therefore cannot release queued work or restore room membership on a replacement connection.

No application realtime action may rely on raw Socket.IO `connect` as an authorization signal. A production probe must be able to perform its very first action immediately after a matching socket-bound `session:ready` and reach canonical chat authorization without receiving the handshake-race sentinel `unauthorized`.

Disconnect is checked both before local authority is installed and again after the asynchronous private-room join. A transport that disappears during admission cannot leave stale in-memory session maps behind or receive authoritative readiness after it is gone.

The web transport treats readiness according to the semantics of each operation:

- messages are durable user intent and wait, with a bounded eight-second authorization timeout, for socket-bound `session:ready`;
- conversation membership is desired state, so active conversations are rejoined exactly once after an authorized reconnect;
- leaving a conversation before readiness cancels that desired membership rather than replaying a stale join;
- typing indicators are ephemeral presence and are dropped while the session is not ready rather than replayed later;
- a token change clears desired room state and forces a fresh transport authorization boundary before new actions are released; and
- expiry, revocation, or explicit logout clears readiness and desired conversation membership.

Concurrent message callers share the same pending readiness boundary. This avoids listener growth and ensures one failed authorization attempt rejects every waiter consistently rather than leaving partially authorized callers behind.

## Passive realtime delivery

Revocation must also prevent an idle socket from continuing to receive private payloads.

Relationship authorization remains the first delivery authority and still produces private `user:<userId>` rooms. Session Authority then evaluates the connected session IDs for those authorized users. Active session rows are selected inside a PostgreSQL transaction with `FOR SHARE`, and delivery executes while those row locks are held. Revoked or expired socket IDs are excluded from the authorized user-room broadcast.

Revocation updates conflict with those share locks, creating deterministic ordering:

- if revocation commits first, the session is absent from later delivery;
- if delivery acquires the session authority first, that already-authorized delivery completes before revocation can commit.

For persisted messages, this recipient filtering transaction begins only after canonical message commit. For typing, it is nested inside the already-open actor authority transaction by reusing the same transaction client rather than opening another transaction.

This preserves the existing block-dominance and relationship-lock ordering while adding session dominance beneath the private-room layer.

## Client logout behavior

Web logout captures the current token, immediately disconnects realtime and clears local auth state, then sends the revocation request with the captured token explicitly. Existing UI call sites therefore keep instant sign-out behavior even when they do not await the returned promise.

Mobile follows the same authority model around SecureStore: it captures the token, clears the device credential, then uses the captured credential for server revocation.

Current-session logout remains locally fail-safe if the network is unavailable or the server already considers the captured token invalid. `logout-all` intentionally reports server failure because a client cannot truthfully claim that other sessions were revoked when the server was unreachable, even though the initiating device has already cleared its local credential.

The web Socket.IO transport clears persisted auth on both `session:expired` and `session:revoked`.

## Deployment compatibility

Tokens issued before Session Authority v1 do not contain `sid`. They intentionally fail closed after this release is deployed. Existing signed-in users therefore need to log in again once after rollout. This avoids inventing unverifiable legacy session state.

No refresh-token protocol is introduced in this release. Access tokens remain finite and use the existing configured JWT lifetime.

## Multi-process semantics

The security authority is database-backed and therefore shared across API processes. A revoked session cannot make a protected HTTP request, perform a new realtime action, reconnect, or remain eligible for private realtime delivery once revocation commits.

Without a shared Socket.IO adapter or revocation pub/sub bus, an idle revoked socket on another process may remain physically connected until it acts, expires, or disconnects normally. Physical connection state is not treated as authorization. Immediate cross-process socket teardown is an optional future operational optimization, not a security dependency of this release.

## Qualification contract

The Session Authority CI lane must prove one exact candidate head and must:

1. permit exactly one owned migration, `20260824233000_add_dogos_auth_sessions`, only as a newly added immutable migration, and reject Prisma schema edits or unrelated migrations;
2. apply the complete migration chain to PostgreSQL;
3. pass read-only Prettier and zero-warning lint across touched API, web, mobile, workflow, and documentation surfaces;
4. enforce privacy-thin `dogos_auth.sessions` storage with canonical TEXT foreign keys;
5. prove login/register mint `sid` and persist the session before returning the token;
6. prove `JwtStrategy` requires persisted session authority;
7. prove `session:ready { socketId }` occurs only after persisted authority, local socket authority, and the private user-room join, and that clients reject readiness from stale transport epochs;
8. prove disconnect-before-admission and disconnect-during-room-join leave no realtime authority behind;
9. prove the web client waits for readiness for durable sends, restores desired room membership after reconnect, drops stale typing, and requires fresh readiness after token replacement;
10. prove concurrent readiness waiters fail together on bounded timeout without releasing message traffic;
11. prove malformed and rate-limited realtime traffic is rejected before persisted-session database work;
12. prove compatible `FOR SHARE` admissions may coexist for one active session while admitted realtime work and current-session revocation serialize on that session row;
13. prove lock-bound canonical realtime work receives and reuses the authority transaction client rather than reacquiring Prisma;
14. prove real message persistence completes with a deliberately constrained one-connection Prisma pool;
15. prove passive delivery and revocation serialize on session rows;
16. prove multi-session `logout-all` uses deterministic row-lock ordering and remains user-scoped;
17. pass API, web, and mobile TypeScript;
18. pass auth, gateway, readiness, chat, admission, relationship, and real PostgreSQL session regressions;
19. build API and web production bundles and the real API Docker image;
20. prove against the production image that the first action immediately after matching socket-bound `session:ready` reaches canonical authorization and never receives the pre-readiness `unauthorized` race;
21. prove concurrent authenticated Socket.IO flood traffic completes canonical work without connection-pool starvation, remains rate-limited by authenticated user across reconnect, and leaves unrelated users in independent buckets;
22. prove current-session logout turns a previously valid token into HTTP `401`;
23. prove an already-ready Socket.IO session emits `session:revoked` and is server-disconnected on its next action after logout;
24. prove a server-issued finite JWT reaches matching socket-bound `session:ready`, then emits `session:expired` and is server-disconnected at expiry; and
25. prove `logout-all` invalidates multiple independently issued sessions while leaving another user's session valid.

Diagnostic heads do not qualify the release.
