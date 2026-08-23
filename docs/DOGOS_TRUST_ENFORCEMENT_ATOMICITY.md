# dogOS Trust Enforcement Atomicity v1

## Goal

Make committed trust-safety state dominate relationship writes at the database ordering boundary.

Before this release, chat access was checked before message persistence began. A block could therefore commit after the access check but before the message row, allowing a message to cross an already-committed block. New direct-conversation creation had the same ordering gap.

## Shared relationship lock

Blocking, unblocking, new direct-conversation creation, and new message persistence now share a symmetric PostgreSQL transaction advisory lock for each unordered user pair.

The key is derived from sorted canonical user IDs. Multi-party message authorization derives one pair lock for every other conversation participant, deduplicates the keys, sorts them, and acquires them sequentially to provide deterministic lock ordering.

The advisory lock query returns a normal integer row around `pg_advisory_xact_lock(...)`; Prisma never has to deserialize PostgreSQL's `void` return type.

## Block dominance

`blockUser()` acquires the relationship lock inside the same transaction that upserts the canonical block row. The lock is held through commit.

A message or new direct-conversation transaction ordered after that commit acquires the same relationship lock and checks the block row only after the lock is obtained. It therefore sees the committed block and rejects the write.

`unblockUser()` uses the same serialization boundary so trust-state transitions have one consistent relationship ordering primitive.

## Message writes

The existing outer conversation-access check remains as an early authorization gate. It is not the authoritative write-time boundary.

For any new canonical message, the persistence transaction:

1. reloads conversation membership;
2. acquires all relationship locks in deterministic order;
3. rechecks block policy;
4. rechecks the delivery receipt;
5. creates the message;
6. creates the idempotency receipt.

Persist-before-emit and the Chat Delivery Integrity client retry contract remain unchanged.

## Direct conversations

PR #19's dedicated direct-pair advisory lock remains in place for duplicate-thread prevention. The relationship lock is acquired after that pair lock and before block lookup. This preserves the proven direct-thread uniqueness mechanism while also serializing conversation creation against trust-state changes.

## Ancillary block effects

The canonical block transaction commits before ancillary effects run:

- pending/accepted meetup cancellation;
- pet-edge `AVOID` updates;
- trust-safety telemetry.

These effects remain important, but failure in an ancillary subsystem must not roll back or delay the canonical safety boundary.

## Qualification

The dedicated Trust Enforcement Atomicity lane must prove:

- no schema or migration ownership;
- the full existing migration chain applies;
- read-only formatting and zero-warning API lint;
- symmetric deterministic relationship lock keys;
- no direct Prisma deserialization of advisory-lock `void`;
- block/unblock acquire the shared lock inside their transactions;
- block commit precedes ancillary cleanup;
- message persistence reacquires and rechecks block policy behind the lock;
- new direct-conversation creation checks block policy behind the same relationship lock;
- a real PostgreSQL race where outer chat access succeeds but a block commits first rejects the message with zero message rows and zero receipts;
- existing direct-thread concurrency, delivery-receipt, gateway, and trust regressions remain green;
- the API production build succeeds.

Triggered inherited workflows remain independently authoritative on the same exact PR head.
