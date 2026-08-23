# dogOS Network Integrity + Scale v1

## Purpose

This release hardens the direct-message network without changing its privacy model or adding a new schema.

It addresses two concrete production seams:

1. simultaneous attempts to start the same direct conversation could race through the previous check-then-create flow and create duplicate threads;
2. Inbox conversation loading performed authorization and unread-count work once per conversation after the initial list query.

## Direct-thread uniqueness under concurrency

`ChatService.createDirectConversation()` now runs the pair lookup and create path inside one Prisma transaction.

Before reading candidate conversations, the transaction acquires a PostgreSQL transaction-scoped advisory lock derived from a symmetric pair key:

`woof:direct-chat:<sorted-user-a>:<sorted-user-b>`

Because the user IDs are sorted before hashing, A → B and B → A serialize on the same lock. The lock is released automatically when the transaction commits or rolls back.

After the lock is held, the existing privacy rules remain authoritative:

- either-direction blocks reject the request;
- an established exact two-person conversation is reused even if the target later becomes non-public;
- a brand-new conversation can only be created for a `PUBLIC` target;
- conversation creation and `CONVERSATION_STARTED` telemetry commit atomically.

This release intentionally uses no schema migration. Advisory-lock hash collisions can only serialize unrelated pairs unnecessarily; they cannot create duplicate conversations or merge identities.

A real PostgreSQL integration contract fires concurrent requests in both directions and requires:

- exactly one response with `created: true`;
- one shared conversation ID across all responses;
- exactly one persisted two-person direct conversation;
- exactly one `CONVERSATION_STARTED` telemetry record.

## Bounded Inbox loading

`ChatService.listConversations()` no longer runs `assertConversationAccess()` and `message.count()` once per thread.

The Inbox path is bounded to three data operations for up to 50 candidate conversations:

1. one conversation query containing participants and latest message;
2. one block-edge query covering every other participant in both directions;
3. one set-based PostgreSQL unread-count query across all visible conversation IDs.

The unread query joins `conversation_participants` to `messages`, respects the caller's `last_read_at`, excludes the caller's own messages, and returns zero for conversations without unread messages.

The privacy semantics are preserved because the initial conversation query already requires the caller to be a participant, only exact two-person threads are surfaced, and the batched block query checks both block directions before any thread is returned.

`getMessages()`, `markRead()`, realtime room joins/leaves, typing, and message persistence continue to use the authoritative `ChatSecurityService.assertConversationAccess()` path.

## Non-goals

This release does not:

- change chat visibility or block policy;
- create group-chat semantics;
- weaken realtime server authorization;
- change idempotent message receipts;
- add a chat migration or uniqueness column;
- expose conversation identifiers through operational metrics.

## Qualification

One exact head must pass every workflow triggered by the changed ownership surface. At minimum:

1. `dogOS Network Integrity + Scale CI`;
2. `dogOS Trust + Discovery CI`;
3. root `CI`.

The dedicated lane verifies formatting/lint/type safety, prohibits schema ownership, structurally requires the symmetric advisory lock and set-based Inbox path, runs unit contracts, runs the real PostgreSQL concurrency/unread integration suite, replays chat-security tests, and builds the API.

Diagnostic heads do not count.
