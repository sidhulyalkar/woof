# dogOS Connectors

Connectors is the external-system boundary for dogOS. It is deliberately split from the canonical dog, activity, care, media, Story, and Autopilot domains so vendor APIs cannot become hidden sources of truth.

## Phase C v1 rule

**A provider transport may authenticate, fetch, normalize, deduplicate, and prove provenance. It may not directly rewrite canonical dogOS records.**

The intended pipeline is:

`provider transport → verified connector identity → normalized import envelope → domain importer → canonical source or immutable observation`

For wearables, the domain importer already exists: Fi/Tractive summaries pass through Autopilot's adapter and immutable zero-reward CareEvent path.

## Provider registry

Phase C begins with these provider classes:

| Provider | Domain | v1 availability | Permitted capability |
| --- | --- | --- | --- |
| Fi | Wearable | `PARTNER_REQUIRED` | Daily activity, device status |
| Tractive | Wearable | `PARTNER_REQUIRED` | Daily activity, device status |
| Veterinary partner | Vet | `PARTNER_REQUIRED` | Appointment/vaccination import, medication/document references |
| Chewy | Retail | `PARTNER_REQUIRED` | Catalog references, user-approved handoff |
| Petco | Retail | `PARTNER_REQUIRED` | Catalog references, user-approved handoff |

`PARTNER_REQUIRED` is intentional. dogOS does not turn a consumer login page, reverse-engineered endpoint, or undocumented mobile API into a claimed OAuth integration.

A future provider can move out of this state only after its supported API contract, credentials, scopes, callback policy, revocation behavior, and signing requirements are verified.

## Credential storage

The pre-existing `IntegrationToken.data` column is ordinary JSON. Existing push-subscription usage demonstrates that the field itself does not provide encryption.

Phase C therefore wraps connector credential payloads in an authenticated encryption envelope before persistence:

- AES-256-GCM
- fresh 96-bit IV for every write
- authenticated tag
- user + provider binding through additional authenticated data (AAD)
- versioned envelope (`v=1`, `alg=A256GCM`)
- no plaintext access or refresh token in the database JSON

`CONNECTOR_CREDENTIALS_KEY` must decode to exactly 32 bytes. Production startup fails when Connectors is enabled without a valid key.

For this first scaffold, `IntegrationToken` is used only as the encrypted **credential vault**. It is not the future home for sync cursors, provider-pet identity mapping, import receipts, or provenance history. Those require explicit relational persistence in the next Phase C slice.

## Wearables

Fi and Tractive retain the Phase A normalization policy:

- daily activity and device-status summaries only
- unrecognized vendor fields are discarded
- location/GPS-shaped payloads are rejected by Autopilot
- imported wearable observations are private and `safetyEligible=false`
- wearable observations cannot earn Bond XP
- connector transport cannot mutate `Pet` or `Activity`

A connector wearable import requires a verified encrypted credential row before it can reach Autopilot.

## Veterinary records

Veterinary transport is provider-specific. Phase C does not pretend generic human FHIR support is equivalent to a veterinary EHR integration.

The registry defines the allowed v1 intent:

- appointments may be imported as sourced records/observations
- vaccination data may be imported with provenance
- medication information is a reference to veterinarian-provided instructions
- documents remain source references
- no connector computes dosage, prescribes treatment, or directly rewrites canonical pet health fields

Provider-specific identity mapping and immutable import receipts are required before a real vet transport can be enabled.

## Retail

Retail is approval-first:

- catalog/product references may inform suggestions
- handoff may occur only after the user chooses to proceed
- no autonomous cart additions
- no autonomous order placement
- no autonomous payment or charge

Chewy and Petco remain partner-gated until a supported integration contract is available.

## Location

Precise tracker location import is disabled in this slice. Provider definitions explicitly report `preciseLocationEnabled=false`.

Before location can be enabled, Connectors must add an explicit scope, retention duration, deletion/revocation behavior, and user-visible permission surface. Phase A's location-rejection behavior remains the default.

## Current API

All routes are JWT-protected and feature-gated by `ENABLE_DOGOS_CONNECTORS`.

- `GET /connectors`
  - provider capabilities
  - truthful connection state
  - credential-vault readiness
  - global safety boundaries
- `POST /connectors/:provider/oauth/start`
  - currently returns an explicit `partner_required` conflict for registry providers
  - never invents an undocumented authorization URL
- `DELETE /connectors/:provider`
  - deletes local encrypted credentials
  - does not falsely claim remote revocation when no official revocation transport is configured
- `POST /connectors/:provider/import/wearable`
  - Fi/Tractive only
  - requires a verified connector credential
  - delegates to Autopilot normalization and reward policy

## Next persistence slice

Before enabling a real external sync, add explicit relational models for:

1. connector connection metadata and lifecycle
2. provider ↔ Woof pet identity mapping
3. per-resource sync cursors
4. immutable import receipts with provider external ID/hash, timestamps, source kind, status, and dedupe uniqueness
5. revocation/deletion receipts

Secrets should remain in the encrypted credential vault, separate from these queryable metadata models.

## CI invariants

The dedicated Connectors lane must prove:

- credential plaintext is absent from persisted envelopes
- ciphertext is bound to user/provider AAD
- missing/malformed encryption key fails closed
- unsupported/partner-gated providers cannot manufacture OAuth state
- a connection state requires an actual encrypted credential row
- wearable import requires connection verification
- wearable import reuses Autopilot
- retail definitions prohibit autonomous purchase
- precise location remains disabled
- connector source contains no direct canonical `Pet`, `Activity`, `CareEvent`, or `MediaAsset` mutation calls
- strict lint/type-check and API build remain green
- inherited Autopilot, Our Story, Foundation, Adventure, and root CI remain green on the same exact head
