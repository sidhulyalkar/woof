# Account Deletion Authority v1

## Purpose

Woof supports account creation, so public beta must also provide a bounded, user-controlled path to permanently delete an account without leaving known Woof-owned identifiers or canonical private Media Library objects behind.

The v1 authority is intentionally server-owned. Clients request deletion; they do not decide which tables, objects, sessions, integrations, pets, or relationship evidence are authoritative.

## API contract

`DELETE /users/me`

- requires the normal authenticated JWT/session boundary;
- derives the deletion subject exclusively from `req.user.sub`;
- does not accept a user ID in the URL or request body;
- returns `{ "deleted": true }` only after the server deletion authority completes.

This endpoint is destructive and is not an administrative delete-by-ID surface.

## Deletion order

Deletion is deliberately ordered around privacy rather than convenience:

1. resolve the authenticated user, owned pets, household memberships, conversations, and canonical private Media Library keys;
2. delete Media Library originals and derivatives from configured private object storage;
3. if private object deletion fails, fail closed before deleting relational account state;
4. enter one PostgreSQL transaction;
5. remove known legacy identifier-bearing rows that do not have foreign keys;
6. detach or delete current rows whose foreign-key policy otherwise blocks user deletion;
7. remove known legacy ML training exports that embed the user's or owned pets' identity inside JSON;
8. delete the canonical `users` row;
9. let modern relational and dogOS operational schemas cascade from user/pet authority;
10. remove relationship containers that became truly empty because of the deletion.

S3-style object deletion and PostgreSQL cannot participate in one atomic transaction. v1 therefore prioritizes avoiding durable private object orphans. Object deletion happens first and is idempotent; a later database failure can leave missing media bytes referenced by metadata until a retry/repair, but it does not leave successfully deleted account metadata while private Media Library bytes remain silently retained.

## Modern cascade authority

The maintained dogOS operational schemas already anchor important identity to canonical user/pet rows with `ON DELETE CASCADE`, including:

- authenticated sessions;
- intelligence observations;
- connector connections;
- discovery location cells;
- chat delivery receipts;
- Companion profiles;
- Social Adventure preferences and related user-owned state;
- caregiver grants.

The account-deletion integration lane applies the full migration chain before testing and exercises representative raw dogOS rows so these are database behaviors rather than schema-reading assumptions.

## Explicit legacy cleanup

Several older beta tables predate the stronger authority model and contain user/pet identifiers without foreign keys. v1 explicitly handles the maintained identity-bearing fields in:

- telemetry;
- meetup proposals;
- co-activity segments;
- service intents;
- legacy gamification, point, badge, and streak tables;
- proactive nudges and cooldowns;
- safety verification and moderation identifiers;
- block relationships;
- legacy `MLTrainingDataPoint` JSON identity.

Current Reward redemption identity is detached, while meetups and community events owned by the deleting account are removed so their restrictive foreign keys cannot preserve the account accidentally.

## Private Media Library guarantee

For canonical `MediaAsset` rows owned by the account or its owned pets, v1 collects and deletes:

- the original `storageKey`;
- every `MediaDerivative.storageKey`.

If configured object storage refuses deletion, account deletion returns a service-unavailable failure and the relational account remains present for retry.

## What v1 does not claim

### Legacy verification document bytes

The historical verification controller constructs `/uploads/documents/...` references, while the verification service explicitly describes durable upload as belonging to a separate storage endpoint. The current repository does not establish a trustworthy object-ownership/deletion contract for those legacy document URLs.

Therefore **v1 does not claim physical deletion of legacy verification-document bytes**. The relational verification rows cascade with the user, but public launch language must not state that every historical verification file is physically erased until that upload/storage path is modernized under explicit private-storage authority.

Before a public workflow invites real identity or vaccination document uploads, Woof should either:

1. migrate verification documents into the canonical private-storage contract with explicit storage keys and deletion ownership; or
2. keep legacy verification-document upload unavailable/out of public-beta scope.

### External-provider retention

Woof deletes its own connector/session records through database authority. A third-party provider may retain data according to its own policies after token revocation/deletion. Provider-specific remote deletion is not inferred unless an integration contract explicitly implements and qualifies it.

### Backup expiration

Online deletion is not a claim that immutable infrastructure backups are instantly rewritten. Production retention and restore policy must document bounded backup retention and ensure deleted identities cannot silently re-enter active production state after restore.

## Qualification

The v1 release lane must prove, on the full migrated PostgreSQL schema:

- authenticated self-only routing;
- private original + derivative object deletion is requested;
- storage failure leaves relational account state intact;
- canonical user and owned pet deletion;
- known no-FK legacy identifier cleanup;
- legacy ML JSON identity cleanup;
- representative raw dogOS session/profile cascade behavior;
- empty household/conversation cleanup without deleting shared containers;
- unrelated peer users remain present;
- API type-check, zero-warning lint, and build.

## Product follow-up

The client UI should land only after this server contract is qualified. The native destructive flow should require an explicit confirmation, explain the irreversible scope in plain language, call `DELETE /users/me`, clear local SecureStore/session state only after server success, and return to the unauthenticated root.
