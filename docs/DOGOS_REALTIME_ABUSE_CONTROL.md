# dogOS Realtime Abuse Control v1

## Release intent

Woof's HTTP API has global throttling, but Socket.IO event handlers do not pass through that HTTP guard. Realtime chat therefore needs an explicit admission boundary before authenticated events can create database writes, conversation-access reads, or relationship-lock work.

This release adds that boundary without changing canonical chat persistence, block policy, message idempotency, realtime recipient authorization, or database schema.

## Admission authority

`RealtimeAdmissionService` is a singleton inside `ChatModule` and keys buckets by authenticated user ID plus event class.

The gateway consumes admission only after socket authentication has established the user identity, and before expensive work:

- `message:send` before `persistMessage`
- `conversation:join` and `conversation:leave` before `assertConversationAccess`
- `typing:start` and `typing:stop` before `withAuthorizedRealtimeRecipients`

A rejected event returns:

- `success: false`
- `error: rate_limited`
- bounded `retryAfterMs`

Rate limiting does not masquerade as an authorization failure and does not mutate canonical chat state.

## v1 policies

Message attempts:

- 5 per 1 second
- 60 per 60 seconds

Typing events, with start and stop sharing one bucket:

- 8 per 5 seconds
- 60 per 60 seconds

Conversation join/leave events, sharing one membership bucket:

- 10 per 5 seconds
- 60 per 60 seconds

The current web client emits typing only on focus and blur, not for every keystroke. These limits therefore leave substantial room above normal product behavior while cutting off automated event floods.

## Reconnect behavior

Admission state is keyed by authenticated user rather than socket ID and is not cleared on disconnect.

Reconnecting the same user to the same API process cannot reset an exhausted bucket. Socket presence state remains separate and is still removed normally on disconnect.

## Bounded memory

The limiter retains at most 10,000 active action/user buckets per API process.

Expired timestamps are pruned periodically. When the cap is reached, stale state is removed first and the least recently seen remaining bucket is evicted if capacity is still required. One authenticated client therefore cannot grow limiter memory without bound by manufacturing socket IDs.

## Scale boundary

Realtime Abuse Control v1 is process-local, matching the current in-memory storage model used by the Nest HTTP throttler.

It guarantees bounded work per API process and prevents reconnect resets on that process. It is not a claim of a globally distributed user quota across arbitrary API machines or regions. If Woof later requires a fleet-wide abuse budget, the admission storage can move to a shared atomic backend while preserving the gateway's admission-first contract and public error semantics.

## Hard boundaries

This release must preserve:

- no database schema or migration changes
- no weakening of PR #20 message retry/idempotency behavior
- no weakening of PR #21 write-time block dominance
- no weakening of PR #23 lock-bound realtime authorization
- no use of socket ID as the rate-limit identity
- no clearing a user's admission state on disconnect
- no database or relationship-lock work before a denied event returns
- no unbounded limiter map
- no per-keystroke requirement in the web client

## Qualification

The dedicated CI lane must prove on one exact head:

1. all inherited migrations still apply to real PostgreSQL;
2. touched-file formatting, zero-warning lint, API TypeScript, unit contracts, and API build pass;
3. message, typing, and membership admission occurs before downstream chat work;
4. the limiter is user-keyed, expiry-pruned, and bounded;
5. prior chat authorization, persistence, retry, and block-dominance regressions remain green;
6. the production Docker image boots;
7. an authenticated Socket.IO client is cut off for message, typing, and membership floods;
8. reconnecting the same user does not reset an exhausted message bucket;
9. a distinct authenticated user retains an independent message bucket.

Earlier diagnostic heads never count as release evidence.
