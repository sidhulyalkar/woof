# dogOS Web Push encrypted subscription authority

Issue: #114

## Security boundary

Web Push subscriptions contain bearer-like delivery material: the provider endpoint plus the browser-generated `p256dh` and `auth` keys. That material is private credential state and must not be stored as plaintext application JSON.

Woof stores Push subscription material in `IntegrationToken.data` as an authenticated AES-256-GCM envelope. The implementation reuses the qualified `ConnectorCryptoService` primitive and `CONNECTOR_CREDENTIALS_KEY`, while Push and connector credentials use separate authenticated-data namespaces.

Push context:

`dogos-push-subscription-v1:<userId>`

Copying a Push envelope into another user or connector context therefore fails authentication even when the same root key is configured. `CONNECTOR_CREDENTIALS_KEY` is a 32-byte base64 key, and production startup fails closed when VAPID keys are configured without a valid encryption key.

## Current Web delivery boundary

The current responsive Web client intentionally registers **no application service worker**. Historical PetPath service-worker authority is being retired, so **background Web Push is intentionally unavailable** in this Web release.

That means the encrypted server-side Push substrate and the current browser product have different maturity states:

- encrypted subscription storage, migration, recipient ownership, cleanup, and revocation remain maintained and repository-qualified;
- the public arbitrary-recipient sender remains retired;
- the current Web settings page does not request notification permission, create a `PushSubscription`, or claim background delivery;
- the old service-worker-dependent Web Push hook remains retired;
- API helpers remain available as dormant transport/recovery primitives, not proof that the current browser UI can subscribe;
- native notification delivery is a separate client authority and is not inferred from this Web Push contract.

A future browser Push release must deliberately reintroduce an application service worker and earn its own lifecycle, privacy, revocation, browser, and live-provider qualification. Backend capability existing today is not permission to resurrect that UI implicitly.

## Runtime server authority

- `GET /api/v1/notifications/subscription` derives ownership from the authenticated session and returns only subscription state plus a SHA-256 subscription fingerprint for a usable server row. It never returns endpoint, `p256dh`, or `auth` material.
- The fingerprint covers canonical endpoint, expiration, `p256dh`, and `auth` material in a fixed JSON shape. Rotated Push keys at an unchanged endpoint therefore produce a different identity.
- `POST /api/v1/notifications/subscribe` derives subscription ownership from the authenticated session. The request body cannot select another `userId`.
- `POST /api/v1/notifications/subscription/revoke` accepts only the authenticated current subscription fingerprint. The server decrypts privately and compare-deletes the exact encrypted snapshot. A mismatch, key rotation, or concurrent replacement is a safe no-op.
- Invalid-row cleanup re-reads the current row, proves the inspected snapshot is still invalid, and compare-deletes that exact snapshot. A valid or concurrently replaced row survives.
- **Provider 404/410 cleanup is bound to the full subscription fingerprint** that actually failed delivery. It cannot account-wide delete a replacement whose endpoint or Push keys changed before cleanup.
- `DELETE /api/v1/notifications/unsubscribe` remains a separate account-wide recovery/revocation path. It removes the authenticated account's Push row without requiring successful decryption, so cleanup still works after corrupt ciphertext or key-loss incidents.
- The old public `POST /api/v1/notifications/send` testing surface remains retired. Internal application services may call `NotificationsService.sendPushNotification` only with server-selected recipients.

These endpoints remain hardened because legacy rows may exist and because future qualified clients may use the substrate. Their existence does not make current Web Push user-facing.

## Current multi-device boundary

`IntegrationToken` has one active server Push row per account because `(userId, provider)` is unique and Push uses `provider=push_subscription`.

This contract does **not** claim multi-device Push fan-out. A future qualified client registering a new subscription can replace the prior account-level row. Full-material fingerprinting plus atomic compare-and-delete prevent stale cleanup from deleting a different or rotated subscription, but they do not create a device registry.

True multi-device storage, per-device revocation, fan-out, and migration remain separate work.

## Encrypted write semantics

Every new or refreshed subscription is encrypted before `IntegrationToken.upsert`.

The stored JSON contains only envelope fields:

- `v`;
- `alg`;
- `iv`;
- `tag`;
- `ciphertext`.

Endpoint, `p256dh`, and `auth` values must not appear as plaintext siblings or application telemetry. Tampered, malformed, wrong-context, or undecryptable envelopes are never reinterpreted as legacy plaintext. Partial envelope-shaped data fails closed as well.

## Legacy plaintext compatibility

Rows written before encryption may contain the historical plaintext shape. Runtime compatibility is intentionally one-way **and time-bounded**.

`PUSH_LEGACY_PLAINTEXT_READS_UNTIL` controls the temporary runtime window:

- empty or absent means runtime plaintext reads are disabled;
- the value must be ISO-8601 with an explicit timezone;
- production startup rejects a cutoff more than 30 days in the future;
- an expired cutoff explicitly disables runtime plaintext reads;
- operators should remove the setting after migration instead of extending it.

Inside an active compatibility window, a valid legacy row may be read only when encryption is configured. Before delivery it is encrypted with the Push-specific context, and migration uses compare-and-swap against the exact JSON snapshot read. A concurrent replacement wins and is never overwritten.

Outside that window, a valid plaintext row becomes `LEGACY_MIGRATION_REQUIRED`. Runtime status treats it as unavailable and delivery fails closed with `legacy_migration_required`. It is not guessed, silently deleted, or downgraded into another shape.

The **explicit migration command is deliberately independent of the runtime compatibility cutoff**. Operators can finish encrypting legacy rows after runtime plaintext reads have been disabled. The scanner advances by monotonically increasing row ID so deleting a previously processed row cannot invalidate the next page.

## Explicit migration command

Run:

`pnpm --filter @woof/api migrate:push-subscriptions`

Optional batch size:

`PUSH_SUBSCRIPTION_MIGRATION_BATCH_SIZE=100`

Valid batch sizes are 1 through 1000. The command requires `DATABASE_URL` and `CONNECTOR_CREDENTIALS_KEY` and emits a counts-only JSON report:

- `scanned`;
- `migrated`;
- `alreadyEncrypted`;
- `invalid`;
- `concurrentChanges`.

It must never print user IDs, row IDs, endpoints, subscription keys, ciphertext, IVs, authentication tags, encryption keys, or arbitrary crypto/provider exceptions. `concurrentChanges` means the compare-and-swap correctly declined to overwrite newer state; re-running is safe.

## Deployment and rollback boundary

This is a data-format migration without a Prisma schema migration.

The pre-encryption application revision is **not data-compatible with encrypted Push rows**. After the first encrypted write or migration, a blind rollback can misclassify envelope data. Prefer roll-forward repair. If rollback to old code is unavoidable, disable Push delivery until an explicitly reviewed data-compatibility plan exists. Never decrypt rows back to plaintext as automatic rollback behavior.

The current Web retirement does not require deleting encrypted server rows opportunistically. Account deletion/recovery authority must continue to remove them safely, and explicit migration remains useful wherever legacy rows still exist.

## Key rotation authority

Push currently shares `CONNECTOR_CREDENTIALS_KEY` with connector credential envelopes, while authenticated-data namespaces prevent cross-context substitution.

**Replacing the environment key in place is not a valid rotation procedure** because existing envelopes would become undecryptable. Rotation needs a separately controlled old-key to new-key migration before retiring the previous key.

The key must never be copied into logs, issue trackers, migration artifacts, source control, or release receipts.

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

The subscription fingerprint is an authenticated control value, not an operational telemetry identifier or URL parameter.

## Repository qualification vs production proof

Repository qualification can prove encrypted storage, wrong-context/tamper rejection, bounded legacy compatibility, compare-and-swap migration, exact cleanup under concurrent replacement, authenticated recipient ownership, privacy-safe diagnostics, and that current Web UI does not claim browser Push delivery.

It **does not prove that production rows were migrated**, that production secrets are configured, that a real provider delivered a notification, that multiple devices are supported, or that the current Web client has background Push authority.

The old production acceptance language requiring a current-browser subscription/delivery check is retired with the browser client path. A future Web Push release must define a new live acceptance contract around its exact service worker, browser lifecycle, permission UX, provider delivery, revocation, account deletion, and rollback behavior before any production delivery claim is restored.
