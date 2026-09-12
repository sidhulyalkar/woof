# Native Social Adventure v1

## Purpose

Native Woof should feel more like a social adventure game without turning a dog into the game piece.

The human gets the competition, collection, discovery, and community feedback. The dog keeps the right to have an ordinary day.

You compete. Your dog does not.

This tranche converges native Community onto the existing server-authoritative Social Adventure system. It does not create a second points economy. Receipt-backed Expeditions now exist as a separate cooperative authority rather than borrowing Social Adventure score.

## Native surfaces

### Community

Community now reads `/social-adventure/feed`, `/social-adventure/me`, and `/social-adventure/leaderboard/global` instead of the legacy `/social/posts` feed.

The native feed renders server-authored semantic moments and the five canonical reactions:

- Nice read
- Good call
- Trying this
- Adventure inspiration
- Cheer

Reactions are culture signals. They do not add league points, become pet labels, or change recommendation authority.

The mobile Social Adventure post type deliberately does not expose legacy like/comment counters or a pet identifier. Native Community only needs the server-approved headline, summary, optional caption, author context, optional pet-name context, and semantic reactions.

Community reads degrade independently. Feed, account score/preferences, and the global league are fetched as separate authority slices. A failure in one optional slice does not erase another slice that the server returned successfully, and a refresh failure preserves previously loaded server-confirmed data rather than manufacturing an empty state.

### Global league

Global visibility remains private by default and changes only after an explicit user press.

Opting in publishes the user's handle and Social Adventure score. It does not publish:

- pet health or symptoms;
- Daily Signals;
- routes or precise location;
- private notes;
- Human Skill practice-score magnitude;
- missed-day or streak state.

The client never derives, sorts, estimates, or repairs rank. It displays the server-issued order and rank.

Preference mutations use the server's mutation response as the immediate authority. Follow-up score and leaderboard reads are opportunistic. If the write succeeds but a later read cannot refresh, native Woof reports that distinction instead of claiming the preference was rolled back.

### Local Packs

Packs are account-level social communities and are available to Guardian and Companion modes. Joining a Pack never grants pet authority.

Pack locality is now a **server-approved structured coarse-region identity**. Web and native clients fetch `/social-adventure/regions` and allow selection only from that catalog. Pack creation and legacy repair send one approved region ID such as `us-ca-south-bay`; the API DTO allowlist and database foreign key independently reject arbitrary locality text.

The v1 region catalog deliberately represents broad public areas at metro, county, or broad-district granularity. It does not require or derive:

- device GPS permission;
- current coordinates;
- home or street address;
- route endpoints or trace history;
- precise venues, schools, apartment complexes, or meetup points;
- reverse geocoding of private coordinates.

This is a stronger claim than the former free-form `regionKey` model, but it remains intentionally bounded. A coarse region is a broad community label, not an anonymity guarantee, proof of residence, or exact physical location. Expanding the catalog is a reviewed server change rather than a fallback to arbitrary text.

#### Legacy locality migration

Historical `regionKey` values cannot safely be assumed coarse because they were user-entered strings. The migration therefore **does not parse, normalize, map, reverse-geocode, or log those values**. Existing LOCAL Pack locality values are discarded before the new foreign key is installed.

Pack identity and membership remain intact. A migrated LOCAL Pack without an approved region becomes `LEGACY_UNVERIFIED`:

- existing members may still see the Pack because membership is server authority;
- it is hidden from nonmember public discovery;
- new joins fail closed;
- local standings fail closed;
- an owner may select one approved region to repair locality;
- changing an already approved locality is not silently allowed through the repair endpoint.

The repair path is deliberately one-way and conservative. It does not expose the discarded legacy text back to clients or telemetry.

Local standings remain fail-closed after locality is approved. The client only renders entries when the server returns `cohortReady: true`. If the server cohort is not ready, native Woof displays the server's privacy message and `memberCount / minimumCohort`. It never calculates whether a cohort should be considered safe itself.

Pack leaderboard responses are also bound to the currently selected Pack. Selection changes invalidate older in-flight requests, stale responses are ignored, and a response whose `pack.id` does not match the requested Pack is hidden rather than displayed under the wrong Pack context.

Pack owners are not offered a fake leave path. Existing server authority requires ownership transfer or retirement before an owner can leave.

## Competition boundary

Social Adventure is a presentation and competition economy for the human side of dogOS. It remains separate from Bond XP, Health Lens, Daily Signals, recommendation authority, the individual dog baseline, and cooperative Expedition receipts.

The current server policy rewards bounded breadth in:

1. Human Skill rooms completed during the weekly season.
2. Distinct eligible Adventure pathways.

It explicitly does not reward repetition volume. CARE does not become competitive Adventure score.

Native Woof must not turn any of the following into score or rank:

- health, symptoms, medication, veterinary care, or CARE completion;
- steps, distance, mileage, duration, speed, exercise intensity, or activity count;
- likes, comments, followers, posting frequency, or reaction volume;
- daily streaks, streak freezes, missed days, or comeback penalties;
- one dog's obedience, fear threshold, behavioral capability, or physical performance;
- Human Skill practice-score magnitude or Marker Timing milliseconds.

A recovery choice, safe stop, easier setup, or quiet day can remain a valid relationship outcome without requiring competitive progress.

## Privacy and integrity

Server authority owns:

- Social Adventure score derivation;
- global opt-in state;
- global rank;
- the approved coarse-region catalog;
- Pack locality validation and legacy locality repair;
- Pack membership;
- Pack cohort readiness and minimum cohort;
- Pack rank;
- block filtering;
- public/private feed visibility;
- semantic reaction persistence.

The native client owns presentation and explicit user intent only.

If an authority slice cannot be loaded, native Woof shows unknown/unavailable state for that slice and may retain an earlier server-confirmed value. It must not substitute cached-looking zeroes, guessed ranks, fabricated privacy state, inferred location, arbitrary locality text, or a response belonging to a different selected Pack.

## Companion mode

Companion users can participate in pet-independent Human Skill, Community, Packs, and the human-side Expedition surface without inventing a dog. These account-level social surfaces do not open Today, Compass, Story, Daily Signals, or any other pet-authorized surface.

A Companion user's Social Adventure score is whatever the server can legitimately derive from that account. The client does not manufacture Adventure evidence to make the league look fuller.

## Relationship to Expeditions

Receipt-backed Expedition Authority now exists independently of Social Adventure score. The canonical Global and Pack Expedition projections are read-only views over immutable, bounded server-issued receipts.

That separation is intentional:

- Social Adventure can expose an optional human-side league and rank.
- Expedition can expose cooperative weekly participation without a podium.
- neither system borrows the other's score, rank, target, or client-derived arithmetic.

Native Community may link to Expedition as another social-game destination, but it must not fold Expedition totals into league score or use league rank to unlock cooperative world state.

The native Expedition presentation remains responsible for honoring server `CALIBRATING` state. Until server targets are calibrated, the client shows descriptive totals without manufacturing completion percentages or reward thresholds.

## Qualification contract

`Native Social Adventure CI` freezes these boundaries by checking that:

- native Community no longer imports the legacy social API;
- feed, score, preference, rank, reaction, and Pack data come from Social Adventure endpoints;
- Community reads use independent settlement so optional league failures do not erase healthy feed authority;
- a successful preference mutation is not described as rolled back merely because a follow-up read failed;
- only the five canonical semantic reactions are exposed;
- global visibility changes through one explicit user action and is never auto-enabled;
- Pack creation and legacy repair use the server-approved coarse-region catalog;
- arbitrary free-form Pack locality normalization is absent from maintained Web/native clients;
- the database owns approved locality through a foreign key, while DTO validation provides an earlier rejection boundary;
- legacy free-form locality is discarded without inference or raw-value telemetry before new locality authority is established;
- unverified legacy Packs cannot be newly joined or ranked until owner repair;
- Pack standings are rendered only behind server `cohortReady`;
- Pack leaderboard responses are request/Pack-bound and stale responses are discarded;
- native Packs contain no device-geolocation implementation;
- client code does not sort or derive league ranks;
- native social types omit pet ID and legacy like/comment counters;
- Expedition remains a separate cooperative authority rather than a Social Adventure score derivative;
- the server Social Adventure score policy tests are rerun;
- the full native client still type-checks and lints with zero warnings.

## Next validation layer

With Social Adventure, structured coarse locality, and receipt-backed Expeditions represented natively, the highest-value validation remains real-world evidence:

1. preserve exact-head release authority and production fail-closed behavior;
2. establish the production API/Web deployment boundary once external credentials exist;
3. rehearse backup/restore against the production-shaped database path;
4. qualify a physical iOS device/TestFlight build;
5. run a small owner and Companion pilot focused on whether Today, Adventure, Skillcraft, Community, Packs, and Expedition feel useful rather than chore-like;
6. observe whether cooperative play improves breadth, learning, recovery choices, and return behavior without increasing volume pressure;
7. calibrate Expedition targets only from real participation evidence, then decide whether completed seasons deserve server-owned Story artifacts or cosmetics.

The next game mechanic should earn its way into the product through observed behavior, not because another counter is easy to add.
