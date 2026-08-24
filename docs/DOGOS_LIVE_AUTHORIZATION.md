# dogOS Live Authorization v1

This release closes a realtime authorization gap without changing database schema, canonical block semantics, conversation membership, or message persistence authority.

## Problem

A Socket.IO client may join a `conversation:<id>` room after an access check. Relationship policy can change after that join. Before this release, message and typing broadcasts targeted the conversation room directly, so an already-connected socket could remain a passive recipient after a block even though new writes were correctly rejected.

Room membership is therefore no longer treated as authorization.

## Delivery authority

Every authenticated socket joins its private `user:<userId>` room at connection time. Conversation rooms may remain for compatibility and explicit join/leave behavior, but production message and typing fanout must never use them as a permission boundary.

Before a realtime event is emitted, `ChatSecurityService.withAuthorizedRealtimeRecipients`:

1. reads the current conversation participants;
2. constructs every unique unordered relationship pair across those participants;
3. acquires the same PostgreSQL transaction-scoped advisory-lock keys used by canonical block/unblock and write-time chat enforcement, in one deterministic global order;
4. re-reads blocked relationships after those locks are held;
5. removes both endpoints of every blocked participant relationship from the realtime recipient set;
6. executes the delivery callback while the authorization transaction and relationship locks are still active.

The callback runs while locks are held deliberately. Returning a recipient list and emitting afterward would reintroduce a check-then-act race where a block could commit between authorization and delivery.

## Persisted messages

Message persistence keeps its existing write-time locked access recheck and idempotent delivery receipt contract.

Realtime fanout is a second, recipient-side policy boundary. If a message was canonically persisted and then a block commits before fanout, the blocked relationship endpoints are excluded from realtime delivery while unaffected participants in a group conversation may still receive the already-accepted message.

If fanout acquires the relationship graph first, the block waits until the delivery callback completes. The event is therefore ordered before the block commit.

## Ephemeral typing events

Typing events have no persisted linearization point. The actor must still appear in the authorized realtime recipient set under the participant relationship locks. If the actor is an endpoint of a currently blocked participant relationship, the typing event is rejected and nothing is emitted.

When authorized, typing fanout targets authorized private user rooms and excludes only the current socket, preserving delivery to another active socket owned by the same user.

## Group conversations

If A and B are participants in a group with C and either A blocks B or B blocks A:

- A is excluded from realtime events for that conversation;
- B is excluded from realtime events for that conversation;
- C remains eligible if C has no blocked relationship with another participant.

This matches the existing conversation-access policy, where a participant cannot access a conversation containing someone with whom they have an active block in either direction.

## Qualification

One exact candidate head must pass every workflow triggered by this ownership surface.

The dedicated Live Authorization lane must remain schema-free and prove:

- all inherited migrations apply to real PostgreSQL;
- strict formatting, lint and API TypeScript checks pass;
- the complete participant relationship lock graph is deterministic;
- message and typing delivery use authorized private user rooms rather than conversation-room fanout;
- blocked endpoints are filtered from group delivery;
- a blocked typing actor cannot emit;
- when a block transaction owns the relationship lock first, realtime delivery waits and then excludes both blocked endpoints;
- when realtime delivery owns the participant relationship graph first, the block transaction waits until delivery completes;
- the prior write-time block-dominance contract remains green;
- the API builds successfully.

Diagnostic heads do not qualify a release.
