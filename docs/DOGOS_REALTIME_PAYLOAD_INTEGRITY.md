# dogOS Realtime Payload Integrity

## Purpose

Socket.IO payload types are untrusted runtime input. TypeScript annotations on `@MessageBody()` do not validate network data, and authenticated traffic must not reach database authorization or relationship-lock work when its shape or size is already invalid.

This release establishes an explicit runtime input boundary for dogOS chat events.

## Transport ceiling

The chat gateway configures Socket.IO with a `32 KiB` maximum packet size through `maxHttpBufferSize`.

The application message contract remains much smaller:

- message text: at most 4,000 raw JavaScript characters;
- conversation identifier: 8–128 ASCII alphanumeric, underscore, or hyphen characters;
- client message identifier: 8–128 ASCII alphanumeric, underscore, or hyphen characters;
- NUL characters are rejected because PostgreSQL text cannot represent them.

The transport ceiling is intentionally above the valid application envelope while remaining far below Socket.IO's default megabyte-scale allowance.

## Ordering contract

For every authenticated realtime chat event, work is ordered as:

1. resolve the authenticated socket user;
2. validate and normalize the payload in process memory;
3. consume the existing authenticated-user admission bucket;
4. perform conversation authorization, relationship locking, persistence, or delivery work.

Malformed input returns `{ success: false, error: 'invalid_payload' }` and does not consume the authenticated user's message, typing, or membership admission budget.

Unauthenticated sockets still fail before payload parsing.

## Persistence defense in depth

`ChatSecurityService.persistMessage()` repeats the shared message contract before its first database access. This protects the persistence seam from future non-Socket.IO callers and makes the zero-database-on-invalid-input property independent of the gateway.

The write-time block-dominance and idempotent receipt contracts remain unchanged after validation succeeds.

## Runtime shapes

The gateway treats all `@MessageBody()` values as `unknown` and explicitly parses:

- `message:send` with `conversationId`, `clientMessageId`, and `text`;
- `conversation:join` and `conversation:leave` with `conversationId`;
- `typing:start` and `typing:stop` with `conversationId`.

`null`, arrays, primitive values, malformed identifiers, empty messages, oversized messages, and NUL-containing messages are rejected without dereferencing unsafe shapes.

## Qualification

The dedicated realtime payload integrity workflow must prove on one exact head SHA:

- shared parser unit contracts;
- gateway validation before admission and authorization work;
- persistence validation before any database call;
- inherited chat authorization and block-ordering regressions on PostgreSQL;
- API type-check, lint, and production build;
- production Docker image boot;
- malformed Socket.IO events return stable `invalid_payload` responses;
- repeated invalid messages do not consume the user's message admission bucket;
- a valid-sized structurally valid message still enters ordinary chat rejection behavior; and
- an over-ceiling Socket.IO frame causes the server to close the connection.

No schema or migration changes are owned by this release.
