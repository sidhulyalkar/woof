# Pet Media Library

Woof's media library is a **private longitudinal record of a pet**, not a public social-media bucket and not a replacement for the owner's primary photo service.

The library exists so behavior, health, coaching, activities, and memories can refer to the same owner-controlled original without duplicating bytes or silently retaining every analyzed image/video.

## Product rules

1. **Analysis is transient by default.** Health Lens and Behavior Vision may analyze media without saving the original.
2. **Keeping an original is explicit.** The owner chooses when a photo/video becomes a durable Woof library asset.
3. **Private is the only default.** Library originals are stored under private object keys and are never served through the public social-media URL.
4. **One original, many meanings.** Albums, health records, behavior observations, coaching sessions, and smart tags reference one asset instead of copying it.
5. **Portable by design.** Owners can download/share originals, export app-created copies to Google Photos, and request a machine-readable manifest.
6. **Provider access is scoped and ephemeral.** Google/Apple/device pickers grant access only to user-selected media. OAuth access tokens must never be written to telemetry or application logs.
7. **Derived AI metadata is provenance-tagged.** An automated behavior/health tag never becomes indistinguishable from an owner tag.

## Current API contract

### Direct private upload

The web/native client requests an upload intent:

```text
POST /media-library/uploads/intents
      ↓
short-lived signed S3/R2 PUT URL
      ↓
client uploads original directly
      ↓
POST /media-library/uploads/complete
      ↓
server HEAD-verifies size + content type
      ↓
asset becomes READY
```

Large videos therefore do not pass through NestJS memory. The signed PUT URL is returned to the authenticated client and is **not persisted**.

### Private reads

`GET /media-library` returns a short-lived signed GET URL for each READY asset. Storage keys are not returned to the browser.

The default signed-read lifetime is 15 minutes. This is long enough for a normal library session but short enough that copied URLs are not durable sharing links.

### Delete

Deleting a library asset deletes the private object and then its library metadata. Features that link to the asset must treat a missing asset as an expected state.

A future `MediaAsset` schema should use soft deletion + a deletion queue when cross-feature references become more numerous. For this v1 contract, hard deletion preserves a simple and understandable owner guarantee.

## Storage namespaces

Recommended object layout:

```text
private/media/<user-id>/<pet-id>/<opaque-object>
private/derivatives/<user-id>/<pet-id>/<asset-id>/<variant>
public/social/<opaque-object>                    # existing social path only
```

Never reuse the public/social CDN URL for private pet-library originals.

### Encryption

Production object storage should enable encryption at rest (provider-managed encryption is the minimum). For veterinary/medical-document vault material, prefer a separately governed bucket or key policy so access policy can evolve independently of ordinary pet memories.

## S3 / R2 CORS

Direct browser uploads and signed reads require bucket CORS. Restrict the origin to the deployed Woof web origins rather than `*`.

Representative policy:

```json
[
  {
    "AllowedOrigins": ["https://app.example.com"],
    "AllowedMethods": ["GET", "HEAD", "PUT"],
    "AllowedHeaders": ["Content-Type", "x-amz-meta-expected-size"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Staging and production should use separate buckets or separate credentials/policies.

## Size and quota defaults

Current safe starting points:

| Type | Default maximum |
| --- | ---: |
| Image | 25 MB |
| Video | 500 MB |
| User private-library quota | 10 GB |

These are configuration values, not product entitlements. Billing plans can later map onto quotas without changing the media contract.

Uploads exceeding declared limits are rejected before a signed intent is created. A completed object whose actual size/content type differs from its declaration is deleted and marked failed.

## Albums

Woof supports two album classes.

### Smart albums

Derived from provenance/tags without moving or copying objects:

- Recent
- Favorites
- Behavior
- Health
- Training
- Adventures
- Imports

### User albums

Owners can create arbitrary private albums such as:

- Puppy year
- Hikes
- Bailey + Nova
- Cooperative care
- Skin follow-up

An asset can belong to multiple custom/smart albums at once.

## Tags and provenance

Tags retain a source:

```text
owner
behavior
health
coach
system
```

That distinction is important. `owner:playful` and `behavior:high-activation` are not equivalent evidence and should not be silently collapsed.

Future ML-derived tags should also carry model/release provenance and confidence through a dedicated derivative record instead of gradually expanding generic JSON.

## Google Photos

Google changed the Photos APIs on **March 31, 2025**. Woof must not depend on historical broad-library Library API scopes.

### Import

Use the Google Photos **Picker API**:

```text
OAuth scope:
https://www.googleapis.com/auth/photospicker.mediaitems.readonly
```

Flow:

1. Browser requests an ephemeral OAuth access token.
2. API creates a Picker session.
3. User selects photos/videos on Google Photos' UI.
4. Woof polls only that session.
5. Server downloads only explicitly selected items.
6. User-selected items are copied into private Woof storage.
7. Picker session is deleted when import completes.
8. OAuth access token is discarded.

Woof does not perform hidden/background full-library sync.

### Export

Woof can upload **app-created items** using:

```text
https://www.googleapis.com/auth/photoslibrary.appendonly
```

The owner selects Woof assets, authorizes the export, and Woof uploads copies to Google Photos.

Do not represent this as arbitrary management of the user's existing Google Photos library. Current Google policy intentionally separates app-created media from the rest of the library.

### Logging rule

Provider routes contain short-lived OAuth credentials in authenticated requests. Request body logging must be disabled/redacted for:

```text
/media-library/providers/*
```

Never include provider bearer tokens in Sentry breadcrumbs, analytics, telemetry, audit text, or exception messages.

## Apple Photos and device libraries

### Web/PWA

Use the browser/system file picker. On iOS this hands selection to the operating-system Photos interface and gives Woof only the selected files.

### Native iOS/iPadOS app

Use **PhotosPicker / PHPicker** as the default import surface. Prefer selection-scoped access over broad PhotoKit authorization.

Use PhotoKit only when a specific future feature genuinely needs persistent library management and can justify that permission to the owner.

Export from the native app should use the operating-system share/save flow so the user remains in control of the destination.

## Importing from other apps

The portable baseline is intentionally provider-agnostic:

- Files / Downloads
- iCloud Drive
- Apple Photos system picker
- Android system photo picker
- Google Photos Picker
- browser drag/drop
- mobile share sheet / document picker

Future providers can implement the same adapter contract:

```text
select explicitly
→ copy chosen original into private Woof storage
→ preserve capture/source metadata where available
→ record provider provenance without storing provider credentials
```

Dropbox, OneDrive, Amazon Photos, NAS/WebDAV, or a camera vendor should not require changes to album semantics.

## Behavior Vision integration

Behavior analysis remains transient by default.

After an analysis, the owner can choose **Keep clip in library**. If chosen, the exact already-captured Blob/File is uploaded as one private asset with:

```text
source = behavior-vision
linkedObservationId = <derived observation id>
tags = behavior + context + phase
```

This creates an explicit durable relationship between the visual evidence and the derived behavior record while keeping unsaved clips ephemeral.

## Health Lens integration

Health Lens follows the same rule:

```text
analyze transiently
→ show result
→ owner chooses whether to keep original privately
```

A future Vet Vault can place selected originals into a stricter encrypted/private sharing workflow without copying them into the social media path.

## Derivative processing roadmap

Originals should remain immutable. Expensive transformations should be asynchronous derivatives:

- thumbnails (AVIF/WebP/JPEG)
- video poster frame
- streaming preview (H.264/HLS where needed)
- media dimensions/duration
- EXIF capture timestamp
- orientation normalization
- perceptual hash for duplicate suggestions
- behavior pose tracks
- visual-health comparison features
- optional owner-approved coarse place/context tags

**GPS EXIF should not be retained or surfaced by default.** If location is useful, convert it to an explicit owner-approved place/context record rather than treating embedded coordinates as harmless metadata.

A production worker should use an idempotency key `(asset_id, derivative_type, processor_version)`.

## Content and security checks

Before a durable upload is considered fully processed:

- verify actual byte size
- verify MIME using file signatures in the derivative worker, not extension alone
- decode image/video in a sandboxed processor
- reject decompression bombs / malformed media
- bound video duration/resolution used for AI analysis
- strip dangerous/irrelevant metadata in derivatives
- never execute container attachments
- scan provider imports under the same policy as direct uploads

The original can remain immutable while access is quarantined until processing succeeds if future threat models require it.

## Data model roadmap

The current v1 metadata implementation uses the existing `Telemetry` event store to avoid destabilizing the beta schema while the product contract is still changing.

That is **not** the forever database design.

Before large-scale production usage, migrate to dedicated indexed entities approximately like:

```text
MediaAsset
MediaAlbum
MediaAlbumAsset
MediaTag
MediaAssetTag
MediaDerivative
MediaExternalReference
MediaExportJob
```

Important indexes:

```text
(owner_id, pet_id, captured_at DESC)
(owner_id, created_at DESC)
(pet_id, media_type, captured_at DESC)
(provider, provider_item_id)
(sha256)
(status, created_at)
```

Telemetry should then record product events about media, not serve as the primary media catalog.

## Backups, retention, and failure recovery

Production storage should enable:

- bucket versioning or equivalent recovery for accidental operational deletes
- lifecycle rules for abandoned PENDING uploads
- multipart-upload expiration
- inventory / orphan-object reconciliation
- database backups
- restore drills
- alerting on storage/API error rates

Abandoned PENDING asset metadata and unclaimed objects should be reconciled by a scheduled cleanup job.

## Product metrics

Do not optimize this library for raw upload volume.

Useful outcomes include:

- % of Behavior/Health sessions where owners intentionally keep evidence
- retrieval of prior media during later decisions
- visual timelines used in vet handoffs
- user-created albums retained over time
- successful imports/exports
- duplicate-storage bytes avoided
- failed/orphaned upload rate
- time-to-first-useful-longitudinal comparison

The library is valuable when it improves memory, coaching, health documentation, and continuity of care, not when it becomes another infinite camera roll.
