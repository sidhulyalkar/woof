# dogOS Realtime Session Lifetime v1

This release makes JWT lifetime an active realtime authorization boundary. It changes no database schema, chat persistence authority, block semantics, rate limits, or product-visible chat behavior beyond ending an expired session.

## Problem closed

HTTP authentication validates JWT expiry on every request. Socket.IO previously validated the JWT only during `handleConnection()`, then stored `sub` in an in-memory `connectedUsers` map for the lifetime of the socket.

That meant a socket established shortly before JWT expiry could remain connected after the token expired. The stale connection could continue attempting message, membership, and typing events, and because it remained in `user:<userId>`, it could also remain eligible for passive realtime delivery.

A signed token without a finite `exp` claim was likewise acceptable to the Socket.IO handshake even though dogOS access tokens are intended to be finite leases.

## Finite session lease

A realtime connection now requires both:

- a non-empty string JWT `sub` claim; and
- a positive, safe-integer JWT NumericDate `exp` claim that is still in the future.

`JwtService.verifyAsync()` remains the signature and standard JWT verification boundary. The gateway additionally requires the expiry claim because realtime authorization must have an explicit end time.

The server stores only the authenticated user ID and expiry timestamp for the active socket. It does not store or log the raw token.

## Active expiry

Each authenticated socket receives a server-side expiry timer tied to the verified JWT deadline.

At expiry the gateway:

1. removes the socket's user and expiry authority from process memory;
2. clears its expiry timer;
3. emits `session:expired` with the bounded reason `token_expired`; and
4. server-disconnects the socket.

Disconnecting removes the stale socket from its private `user:<userId>` room, so an expired connection cannot remain a passive realtime recipient.

Node timers have a maximum single delay. Tokens whose expiry is farther away than that limit are handled by a bounded timer that re-arms until the actual JWT deadline rather than overflowing into an early or immediate timeout.

Normal disconnect cleanup clears the stored session and timer, preventing timer retention after a socket is gone.

## Event-ingress recheck

Timer delivery is not treated as the sole security boundary. Every authenticated Socket.IO action still begins by resolving `connectedUsers.get(client.id)`, and now also verifies that the stored JWT deadline remains in the future before payload parsing, admission accounting, database authorization, relationship locking, persistence, or delivery work.

This closes the event-loop edge where the wall clock has crossed expiry but the scheduled timer has not yet received a turn.

The established ordering becomes:

1. resolve the authenticated socket identity;
2. require an active finite JWT lease;
3. validate and normalize the event payload;
4. consume the existing authenticated-user admission budget;
5. perform canonical authorization, locking, persistence, or delivery work.

The prior payload-integrity, abuse-control, block-dominance, live-recipient authorization, idempotent delivery, and private-user-room contracts remain unchanged.

## Web behavior

The web transport listens for the explicit `session:expired` event and clears the persisted Woof auth state through the existing auth-store logout boundary. A later login can reuse the singleton Socket.IO client because `connectSocket()` always refreshes `socket.auth` from the current auth store before connecting.

This release does not introduce refresh tokens, a server-side token revocation table, or a new session database. It enforces the lifetime already encoded in the signed access token. Explicit server-side revocation is a separate authority problem and should not be implied by this release.

## Privacy

Operational logging remains identifier-free at this boundary. Rejected connections log only the error class name already used by the gateway. Raw JWTs, user IDs, socket auth objects, and token claims are not logged.

## Qualification contract

The dedicated Realtime Session Lifetime lane must prove one exact candidate head and must:

1. reject schema and migration ownership;
2. apply the full inherited migration chain;
3. pass touched-file Prettier and zero-warning API/web lint;
4. require finite future `exp` at the realtime handshake;
5. require session-lifetime validation before payload/admission/chat work for message, membership, and typing handlers;
6. prove normal disconnect clears expiry state;
7. prove an expired lease rejects event work even if its timer has not run yet;
8. pass API and web TypeScript;
9. pass the focused gateway regression suite;
10. build API and web production bundles;
11. build and boot the real production API image; and
12. use a real Socket.IO client against that image to prove a short-lived signed JWT receives `session:expired` and is server-disconnected at its deadline.

Diagnostic heads do not qualify the release.
