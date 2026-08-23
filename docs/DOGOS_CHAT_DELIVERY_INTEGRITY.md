# dogOS Chat Delivery Integrity v1

## Goal

Make an uncertain client retry idempotent end to end.

The server already persists a `(user_id, client_message_id)` delivery receipt and suppresses duplicate realtime emission. This release closes the client-side half of that contract so a lost or delayed Socket.IO ACK cannot turn a user retry into a second canonical message.

## Client send identity

The socket transport does not create `clientMessageId` values.

The chat composer owns a `ChatSendAttempt` containing:

- `conversationId`
- normalized message text
- `clientMessageId`

The first send creates the ID. If delivery is not confirmed and the user retries the same unchanged draft in the same conversation, the exact same ID is reused. The existing server receipt therefore resolves the retry to the already-persisted canonical message instead of creating another one.

Editing the draft invalidates the uncertain attempt identity. A changed draft is a new send intent and receives a new ID.

## ACK semantics

A transport timeout is not proof that the server failed to save the message. The UI therefore says delivery was not confirmed instead of claiming the message was not saved.

After an acknowledged send, the composer clears the draft only when the current input still matches the text that was sent. Text typed while the request was in flight is preserved.

Conversation-list cache refresh is non-authoritative UI maintenance. A cache invalidation failure cannot turn a successfully acknowledged canonical save into a false send failure.

## Existing server authority

This release does not change the database schema or server source of truth.

Server guarantees remain:

- messages are persisted before realtime emission
- `(user_id, client_message_id)` is unique
- a duplicate receipt returns the existing canonical message
- a reused client ID cannot be moved to a different conversation
- duplicate retries do not emit a second `message:received` event
- conversation membership and block policy remain server-authorized

## Qualification

The dedicated Chat Delivery Integrity lane must prove:

- no schema or migration ownership
- read-only formatting and zero-warning lint
- socket transport cannot generate message IDs
- `clientMessageId` is required by `sendMessage`
- the composer owns and reuses uncertain retry identity
- draft edits invalidate retry identity
- late ACK reconciliation preserves newer draft text
- web TypeScript and focused delivery-state tests pass
- existing API receipt and gateway regressions pass
- web and API production builds pass

Trust + Discovery CI and root CI remain independently authoritative whenever their owned paths trigger.
