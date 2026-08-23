# dogOS Trust + Discovery v1

## Release intent

This release turns Woof's network loop into a deployable product path without weakening the canonical dogOS spine.

The user-facing loop is deliberately narrow:

1. choose the active dog
2. review explainable compatibility matches
3. optionally add coarse nearby context
4. start a private direct conversation
5. suggest a public-place meetup
6. accept or decline the meetup
7. report a tiny real-world outcome
8. feed that outcome back into future compatibility evidence

The release does not introduce a second social graph, a second meetup store, a public dog map, or an autonomous agent.

## Canonical and operational ownership

Canonical source-of-truth data remains in the existing Prisma/public domain:

- `Conversation`
- `ConversationParticipant`
- `Message`
- `MeetupProposal`
- `PetEdge`
- `Telemetry`
- `BlockedUser`

Two narrow operational schemas support delivery and privacy mechanics without becoming new product truth:

- `dogos_chat.message_receipts` stores message idempotency receipts.
- `dogos_discovery.locations` stores opt-in coarse discovery cells.

## Realtime trust boundary

Socket authentication alone is not authorization.

Every realtime operation now checks current server-side conversation membership and block state:

- join
- leave
- send
- typing start
- typing stop

Messages are emitted only after the canonical `Message` row has committed.

Every send carries a client-generated `clientMessageId`. A receipt is unique per user and message ID and is bound to one conversation. Retrying the same accepted send returns the persisted message without emitting a duplicate.

Global online/offline broadcasts are intentionally absent. Presence is not product truth and must not leak the network's membership graph.

## Canonical Inbox

The production web Inbox no longer contains seeded conversations or seeded messages.

It reads:

- `GET /chat/conversations`
- `GET /chat/conversations/:id/messages`
- `POST /chat/conversations/:id/read`

Direct conversations are created or reused through:

- `POST /chat/conversations`

The server, not the client, decides whether the relationship is eligible.

## Privacy-safe nearby discovery

Nearby discovery is explicit opt-in.

The browser may submit a latitude/longitude to `PUT /discovery/location`, but the service immediately transforms it into a coarse 0.02-degree cell. The precise coordinate is never written to PostgreSQL and is never included in telemetry.

Operational location records:

- are approximately 2.2 km precision before latitude effects
- expire after 30 days
- can be explicitly disabled
- are never returned as coordinates

Candidate responses expose only one of:

- `WITHIN_2_5_KM`
- `WITHIN_5_KM`
- `WITHIN_10_KM`

They do not expose another household's coordinate, bucket, address, or home location.

Candidate lookup excludes blocked users and non-public profiles before returning any dog.

### Why v1 does not require PostGIS

The repository's shared CI PostgreSQL image is currently pgvector-based and does not guarantee PostGIS. Trust + Discovery v1 therefore uses indexed coarse integer cells so the privacy contract can ship without changing every database environment.

The internal candidate generator can later migrate to PostGIS (`geography(Point, 4326)`, GiST, `ST_DWithin`) while preserving the public contract: explicit consent, no public coordinates, bounded radius, and coarse distance bands.

## Compatibility remains explainable

Nearby context does not replace compatibility scoring.

Existing compatibility behavior remains authoritative:

- deterministic baseline remains available
- learned scoring remains `off`, `shadow`, or explicitly `promoted`
- blocked, avoided, and private relationships remain filtered
- model failure falls back rather than taking discovery down

The web currently uses the privacy-safe nearby set to prioritize and annotate existing explainable compatibility matches. It does not create PetEdges merely because a candidate was viewed.

## Meetup and outcome reuse

The existing `MeetupProposal` source remains canonical.

The release adds three structured, low-friction outcome fields to the existing completion contract:

- dog experience: `loved_it`, `comfortable`, `not_their_thing`
- owner experience: `great`, `fine`, `a_lot_today`
- meet again: `yes`, `maybe`, `no`

The existing safety checklist remains explicit.

Structured answers are also normalized into the existing `feedbackTags` channel (`dog_comfortable`, `owner_great`, `meet_again_yes`) so compatibility dataset generation can consume them without a parallel feedback schema.

## No fake network state

Production network surfaces must not substitute demo data when canonical reads fail.

This release removes:

- hard-coded Inbox conversations
- hard-coded chat messages
- fake coordinate-bearing meetup cards
- hard-coded San Francisco service businesses and ratings
- fake map-marker simulation

Empty and degraded states remain visibly empty or degraded.

## Hard boundaries

Trust + Discovery v1 must preserve all of the following:

- no public member or dog coordinates
- no home-location exposure
- no precise discovery coordinate persistence
- no precise-coordinate telemetry
- no chat room access based only on caller-supplied IDs
- no global presence broadcasts
- no message emit before persistence
- no duplicate sends on idempotent retry
- no meetup proposal before a real two-person conversation with messages
- no coordination across a block
- no fabricated network profiles, conversations, businesses, reviews, or locations
- no new medical inference
- no automatic location consent

## Release qualification

A release candidate is not qualified until one exact head passes:

1. dogOS Trust + Discovery CI
2. dogOS Release Polish CI
3. dogOS Concierge CI
4. dogOS Connectors CI
5. dogOS Our Story CI
6. dogOS Autopilot CI
7. dogOS Foundation CI
8. Adventure System CI
9. root CI, including migrations, backend tests, Chromium contracts, and production API/web builds

Earlier diagnostic heads never count as release evidence.
