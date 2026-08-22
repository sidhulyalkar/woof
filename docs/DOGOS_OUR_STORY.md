# dogOS Our Story

Our Story is the longitudinal life view for a dog and household. It is intentionally a **read-through composition layer**, not a second database of copied activities, care events, media, or tracker data.

## Product thesis

A good dog life is made from ordinary things: walks, play, rest, training, safe opt-outs, photos worth keeping, and years of accumulated rhythm. Our Story turns those existing records into one calm chronology without requiring the user to post, prove, or duplicate anything.

The stable web route remains `/journey` for compatibility, but the product concept and primary navigation label are **Story**.

## Authoritative sources

Our Story reads from:

1. `Activity`
   - one real activity stays one canonical activity
   - multi-pet participants remain attached to that one source record
2. `CareEvent`
   - Adventure reflections, recovery choices, and other care context
   - private events remain private
   - household-visible events can appear to active members who share the pet
3. `MediaAsset`
   - READY private Media Library assets owned by the viewer
   - capture time is preferred over upload time when available
4. bounded wearable context
   - Autopilot daily activity summaries are already immutable zero-reward CareEvents
   - device-maintenance events such as low battery are not life-story moments

Our Story never rewrites these sources.

## De-duplication rules

Activity completion can also emit a CareEvent for Adventure reward integrity. The Story composer suppresses any CareEvent whose context references `activityId`, because the canonical Activity is already on the timeline.

`TRACKER_DEVICE_STATUS` is excluded. `TRACKER_DAILY_ACTIVITY` can appear as context with explicit non-diagnostic wording.

## Curation without copied truth

Story v1 does not require a new polymorphic database table. Owner curation is encapsulated in an internal `STORY_CURATION` envelope using the existing user-owned JSON record store.

A curation envelope contains only:

- source type: Activity, CareEvent, or Media
- source ID
- state: SAVED or HIDDEN
- optional owner note
- curation update timestamp

It does **not** copy source title, timestamps, metrics, route data, health context, media metadata, or other source truth.

Curation writes:

- validate access to the referenced source first
- serialize with a PostgreSQL advisory transaction lock keyed by user + source
- update one existing envelope or create one
- delete only the curation envelope for CLEAR
- never mutate/delete Activity, CareEvent, or MediaAsset records

## Privacy model

### Activity

Story reuses the dogOS household activity read boundary. Household members may see shared activities they already have access to.

### CareEvent

A viewer sees:

- their own CareEvents
- another member's event only when the event is explicitly `HOUSEHOLD` visible and the pet is in a household the viewer actively belongs to

Another member's `PRIVATE` CareEvent never becomes visible simply because the pet is shared.

### Media

Story v1 includes only READY Media Library assets owned by the viewer. A shared dog does not automatically grant access to another person's private photos or videos.

### Location

Story does not expose or infer a raw location history. Life-stat "named places" counts only semantic route labels already attached to activities, such as a place ID/name or named start location. Raw coordinate arrays do not count as a place and are not rendered by Story.

## Life stats

Story derives:

- activity count
- recorded active minutes/hours
- distance when normalized distance metrics are already available
- READY private memory count
- count of semantic named places

The v1 aggregate scan is capped at 5,000 activities. If the canonical activity count exceeds that bound, the API returns `coverage: BOUNDED` and the UI labels the figures as a recent-history estimate rather than claiming completeness.

## Milestones

Derived milestones currently include:

- first recorded adventure
- 10 / 50 / 100 shared activities
- 10 / 50 / 100 recorded activity hours
- first kept private media memory

The first-memory milestone uses the earliest real READY media timestamp, preferring capture chronology where present. No synthetic dates are generated.

## Suggestions and owner authority

Story may mark deterministic moments such as a favorite media item, hike, meetup, new-place context, or safe opt-out as `Worth remembering?`.

This is a suggestion only. The owner decides whether to save, annotate, hide, or leave the moment alone.

Future AI-assisted memory suggestions must follow the same rule:

- AI may propose
- the owner curates
- AI cannot publish, rewrite source facts, or convert a suggestion into a persistent life claim without confirmation

## Feature flag and rollback

`ENABLE_DOGOS_OUR_STORY` gates the API surface.

Outside production, the surface defaults on unless explicitly disabled. In production it disappears with a 404 unless explicitly enabled.

Rollback therefore does not require reverting Activity, Adventure, Autopilot, or Media Library state. Disabling Story removes the composition surface while leaving all authoritative source data intact.

## Qualification invariants

The dedicated Story CI lane must prove:

- inherited migration chain still deploys and Prisma drift stays zero
- exact Story release surface formats cleanly
- zero-warning API and web lint
- API/web type-check
- Story service contracts
- web API contracts and full web tests
- API + web production builds
- Story code does not create/update/delete/upsert canonical Activity, CareEvent, or MediaAsset rows

Root CI and inherited Autopilot, dogOS Foundation, and Adventure lanes remain mandatory on the exact same PR head.
