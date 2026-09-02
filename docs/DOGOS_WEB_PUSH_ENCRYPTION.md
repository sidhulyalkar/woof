# dogOS Web Push encrypted subscription authority

Issue: #114

## Security boundary

Web Push subscriptions contain bearer-like delivery material: the provider endpoint plus the browser-generated `p256dh` and `auth` keys. They are private credential material and must not be stored as plaintext application JSON.

Woof stores Web Push subscription material in `IntegrationToken.data` as an authenticated AES-256-GCM envelope. The implementation reuses the already-qualified `ConnectorCryptoService` primitive and `CONNECTOR_CREDENTIALS_KEY`, but Push and connector credentials use different authenticated-data namespaces.

Push context:

`dogos-push-subscription-v1:<userId>`

Connector contexts remain under their own `dogos-connector-credential-v1:*` namespace. Copying a Push envelope to a different user or connector context therefore fails authentication even when the same root key is configured.

`CONNECTOR_CREDENTIALS_KEY` is a 32-byte base64 key. Production startup fails closed when VAPID keys are configured without a valid encryption key.

## Runtime authority

- `GET /api/v1/notifications/subscription` derives ownership from the authenticated session and returns only subscription state plus a SHA-256 subscription fingerprint for a usable server row. It never returns the endpoint, `p256dh`, or `auth` material.
- The fingerprint covers canonical endpoint, expiration, `p256dh`, and `auth` material in a fixed JSON shape. Rotated Push keys at an unchanged endpoint therefore produce a different identity.
- The browser fingerprints its own full local subscription and considers Push enabled only when that fingerprint matches the encrypted server row.
- A browser with no local subscription, or a fingerprint mismatch, marks itself disabled. Passive status reconciliation never deletes server state because that row may represent another browser/device.
- `POST /api/v1/notifications/subscribe` derives the subscription owner from the authenticated session. The request body cannot select another `userId`.
- `POST /api/v1/notifications/subscription/revoke` is current-browser revocation. The authenticated body contains only the base64url SHA-256 subscription fingerprint. The server decrypts the current row privately and deletes with a compare-and-delete predicate bound to the exact encrypted JSON snapshot. A mismatch, key rotation, or concurrent row replacement is a safe no-op. POST is intentional here so the fingerprint stays out of the URL and does not depend on DELETE-body handling by intermediaries.
- Ambiguous browser subscribe failures use that same current-browser conditional revocation. They never call the account-wide delete as compensation.
- Invalid-row cleanup re-reads the current row, proves that current snapshot is still invalid, and compare-deletes that exact JSON snapshot. A valid or concurrently replaced row survives.
- Provider 404/410 cleanup is bound to the full subscription fingerprint that actually failed delivery. It cannot account-wide delete a replacement whose endpoint or Push keys changed before cleanup.
- `DELETE /api/v1/notifications/unsubscribe` is the separate account-wide recovery/revocation path. It removes the authenticated account's Push row without reading or decrypting it, so deletion remains possible for corrupt ciphertext or key-loss incidents.
- The old public `POST /api/v1/notifications/send` testing surface is retired. Internal application services may still call `NotificationsService.sendPushNotification` with server-selected recipients.
- The Web client exposes status, subscribe, current-browser unsubscribe, and an account-recovery API helper, but ordinary browser lifecycle code never uses account-wide revocation.
- For ordinary browser disable, matching server revocation occurs before local browser unsubscribe. A local cleanup failure therefore cannot restore server delivery authority.

### Current multi-device boundary

The existing `IntegrationToken` authority has one active server Push row per account because `(userId, provider)` is unique and Web Push uses `provider=push_subscription`.

This release does **not** claim multi-device Push fan-out. Registering a new browser can replace the prior account-level Push row. The full-material fingerprint handshake plus atomic current-browser compare-and-delete prevent a different or rotated browser subscription from falsely showing itself subscribed or being deleted by stale cleanup, but they do not turn the singleton persistence model into a device registry.

An identical concurrent registration of the exact same browser subscription is intentionally treated as the same delivery authority. Distinct per-registration attempt identity is not modeled in this singleton design.

True multi-device subscription storage, per-device revocation, fan-out, and migration are tracked separately in issue #119. The privacy fix in this release should not be coupled to a schema redesign under an exhausted CI/deployment evidence budget.

## Encrypted write semantics

Every new or refreshed subscription is encrypted before `IntegrationToken.upsert`.

The stored JSON contains only the envelope fields:

- `v`;
- `alg`;
- `iv`;
- `tag`;
- `ciphertext`.

The endpoint, `p256dh`, and `auth` values must not appear as plaintext siblings in the stored row or in application telemetry.

Tampered, malformed, wrong-context, or undecryptable envelopes are never reinterpreted as legacy plaintext. A partial envelope shape also fails closed even if plaintext-looking subscription fields are present. Invalid rows may be removed only through exact-snapshot cleanup so a concurrent valid replacement is not erased.

## Legacy plaintext compatibility

Rows written before this release may still contain the historical plaintext shape. Runtime compatibility is intentionally one-way **and time-bounded**.

`PUSH_LEGACY_PLAINTEXT_READS_UNTIL` controls the temporary runtime window:

- empty or absent means runtime plaintext reads are disabled;
- the value must be an ISO-8601 timestamp with an explicit timezone;
- production startup rejects a cutoff more than 30 days in the future;
- an already-expired cutoff is valid and explicitly disables runtime plaintext reads;
- operators should remove the setting after migration instead of extending it.

Inside an active compatibility window:

1. a valid legacy row may be read only when the encryption key is configured;
2. before delivery it is encrypted with the Push-specific context;
3. the migration write is a compare-and-swap against the exact JSON snapshot that was read;
4. if a concurrent browser re-subscription changes the row first, the migration does not overwrite it and the current row is re-read;
5. malformed legacy rows are not guessed or repaired from partial credential data.

Outside that window, a valid plaintext row becomes `LEGACY_MIGRATION_REQUIRED`. Runtime status reports it as unsubscribed and delivery fails closed with `legacy_migration_required`. The row is not classified as corrupt, is not conditionally revoked using its plaintext material, and is not deleted by invalid-row cleanup.

The explicit migration command is deliberately independent of the runtime compatibility cutoff. Operators can therefore finish encrypting legacy rows after plaintext runtime reads have been shut off. This prevents extending compatibility merely to complete migration.

The explicit scanner advances by monotonically increasing row ID (`id > lastSeenId`) rather than a Prisma cursor that requires the previous page-tail row to keep existing. A concurrent unsubscribe can therefore delete a processed row without invalidating the next migration query. Rows inserted behind the current high-water mark are safely picked up by a later idempotent run.

## Explicit migration command

Operators may migrate legacy rows independently of normal traffic:

`pnpm --filter @woof/api migrate:push-subscriptions`

Optional batch size:

`PUSH_SUBSCRIPTION_MIGRATION_BATCH_SIZE=100`

Valid batch sizes are 1 through 1000. The command requires `DATABASE_URL` through the normal database package and `CONNECTOR_CREDENTIALS_KEY` for encryption. It remains authorized after `PUSH_LEGACY_PLAINTEXT_READS_UNTIL` has expired or been removed.

The command emits one JSON report containing counts only:

- `scanned`;
- `migrated`;
- `alreadyEncrypted`;
- `invalid`;
- `concurrentChanges`.

It must never print user IDs, row IDs, endpoints, subscription keys, ciphertext, IVs, authentication tags, encryption keys, or arbitrary crypto/provider exception details.

A `concurrentChanges` count is not an error by itself. It means a row changed between read and compare-and-swap, so the migrator deliberately declined to overwrite newer state. Re-running the migration is safe.

## Deployment and rollback boundary

This is a data-format migration without a Prisma schema migration.

The pre-encryption application revision is **not data-compatible with encrypted Push rows**. Its legacy parser does not understand the envelope. After the first encrypted subscription write or migration, a blind code rollback can misclassify encrypted rows as invalid and may remove them during delivery attempts.

Therefore:

- deploy the encryption-capable application before running the explicit migration;
- if temporary runtime compatibility is required, choose a short `PUSH_LEGACY_PLAINTEXT_READS_UNTIL` cutoff no more than 30 days ahead;
- do not run the migration from an older application revision;
- after any encrypted write occurs, prefer roll-forward repair rather than reverting to a pre-encryption revision;
- if an emergency rollback to old code is unavoidable, Push delivery should be disabled until an explicitly reviewed data-compatibility plan is executed;
- never decrypt rows back to plaintext as an automatic rollback behavior.

Database backup/restore evidence belongs to operational-resilience issue #100 and is not implied by this repository migration contract.

## Key rotation authority

Push currently shares `CONNECTOR_CREDENTIALS_KEY` with connector credential envelopes. AAD namespaces prevent cross-context substitution, but compromise or rotation of the root key affects both domains.

Replacing the environment key in place is **not** a valid rotation procedure because existing envelopes would become undecryptable. Rotation requires a separately controlled migration that can decrypt with the old key and re-encrypt with the new key before the old key is retired. Until that procedure is implemented and rehearsed, operators must treat this key as one coordinated integration-vault rotation boundary.

The key must never be copied into logs, issue trackers, migration artifacts, or source control.

## Privacy and telemetry

Application logs may contain bounded state classes and provider HTTP status codes needed for cleanup. They must not contain:

- user IDs;
- endpoint URLs;
- subscription fingerprints;
- `p256dh` or `auth` keys;
- notification titles/bodies;
- ciphertext, IVs, or authentication tags;
- encryption key material;
- arbitrary provider or crypto exception messages/stacks.

The subscription fingerprint is returned only to the authenticated client to compare its own local subscription identity and can be submitted back in the authenticated current-browser revocation body; it is not an operational telemetry identifier or URL parameter.

Browser Push helpers likewise log stable failure classes rather than raw exception objects.

## Repository qualification vs production proof

Repository qualification can prove the source contract, encryption/decryption behavior, wrong-context/tamper rejection, bounded legacy compatibility, legacy compare-and-swap migration, explicit migration after runtime cutoff, deletion-safe batch scanning, session-owned controller authority, full-subscription-fingerprint reconciliation, atomic current-browser compare-and-delete, exact invalid/provider-expiry cleanup, provider privacy behavior, and bounded migration reporting.

It does **not** prove that production rows were migrated, that production secrets are configured, that a real browser granted permission, that multiple devices are supported, or that a Push provider delivered a notification.

Production promotion requires an observed migration report with counts only, the deployed release SHA, target environment identity, successful browser subscription/delivery/revocation checks, and an explicitly reviewed rollback/rotation procedure. None of those live claims should be inferred from CI.
