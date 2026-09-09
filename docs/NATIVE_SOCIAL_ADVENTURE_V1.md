# Native Social Adventure v1

## Purpose

Native Woof should feel more like a social adventure game without turning a dog into the game piece.

The human gets the competition, collection, discovery, and community feedback. The dog keeps the right to have an ordinary day.

This tranche converges native Community onto the existing server-authoritative Social Adventure system. It does not create a second points economy and it does not invent an Expedition system before server authority exists.

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

A Pack currently uses a user-supplied broad-area `regionKey`, such as `south-bay-ca`. Native Woof does not request device location to create or rank a Pack, and it does not derive locality from a home address, route endpoint, GPS trace, or meetup location.

The v1 locality boundary is intentionally stated narrowly: the server validates `regionKey` slug syntax and length, while the native client asks the user for a broad-area label and does not implement device geolocation. v1 does **not** semantically prove that an arbitrary user-entered slug is geographically coarse. Product policy prohibits submitting a street address, precise venue, coordinate, route, or exact meetup point, but stronger structured-region authority is a separate follow-up rather than an unearned privacy claim.

Local standings are fail-closed. The client only renders entries when the server returns `cohortReady: true`. If the server cohort is not ready, native Woof displays the server's privacy message and `memberCount / minimumCohort`. It never calculates whether a cohort should be considered safe itself.

Pack leaderboard responses are also bound to the currently selected Pack. Selection changes invalidate older in-flight requests, stale responses are ignored, and a response whose `pack.id` does not match the requested Pack is hidden rather than displayed under the wrong Pack context.

Pack owners are not offered a fake leave path. Existing server authority requires ownership transfer or retirement before an owner can leave.

## Competition boundary

Social Adventure is a presentation and competition economy for the human side of dogOS. It remains separate from Bond XP, Health Lens, Daily Signals, recommendation authority, and the individual dog baseline.

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
- Pack membership;
- Pack cohort readiness and minimum cohort;
- Pack rank;
- block filtering;
- public/private feed visibility;
- semantic reaction persistence.

The native client owns presentation and explicit user intent only.

If an authority slice cannot be loaded, native Woof shows unknown/unavailable state for that slice and may retain an earlier server-confirmed value. It must not substitute cached-looking zeroes, guessed ranks, fabricated privacy state, inferred location, or a response belonging to a different selected Pack.

## Companion mode

Companion users can participate in pet-independent Human Skill, Community, and Packs without inventing a dog. These account-level social surfaces do not open Today, Compass, Story, Daily Signals, or any other pet-authorized surface.

A Companion user's Social Adventure score is whatever the server can legitimately derive from that account. The client does not manufacture Adventure evidence to make the league look fuller.

## Why Expeditions are not in this tranche

The product direction includes cooperative seasonal Expeditions and Pack goals, but there is not yet a canonical Expedition API or receipt model in the server.

Native v1 therefore stops at Feed + global league + Packs + privacy-safe Pack standings. Shipping a decorative client-side Expedition counter would create exactly the kind of unowned authority dogOS is designed to avoid.

A later Expedition tranche should add server-owned templates, participation receipts, category caps, duplicate protection, recipient/pet eligibility where relevant, and cooperative progress before native presentation is built.

Expeditions are a candidate product layer, not the automatic next release. After this native convergence, production deployment, physical-device use, restore evidence, and a small owner pilot have higher information value. Cooperative mechanics should be promoted when real usage shows that shared Pack goals solve a user problem rather than because the repository can support another subsystem.

## Qualification contract

`Native Social Adventure CI` freezes these boundaries by checking that:

- native Community no longer imports the legacy social API;
- feed, score, preference, rank, reaction, and Pack data come from Social Adventure endpoints;
- Community reads use independent settlement so optional league failures do not erase healthy feed authority;
- a successful preference mutation is not described as rolled back merely because a follow-up read failed;
- only the five canonical semantic reactions are exposed;
- global visibility changes through one explicit user action and is never auto-enabled;
- Pack standings are rendered only behind server `cohortReady`;
- Pack leaderboard responses are request/Pack-bound and stale responses are discarded;
- native Packs contain no device-geolocation implementation;
- client code does not sort or derive league ranks;
- native social types omit pet ID and legacy like/comment counters;
- the locality contract admits that v1 validates slug shape rather than semantically proving geographic coarseness;
- the server Social Adventure score policy tests are rerun;
- the full native client still type-checks and lints with zero warnings.

## Next validation layer

After this convergence qualifies, the highest-value work is to move Woof from repository confidence to real-world evidence:

1. protect `main` and preserve exact-head release authority;
2. establish the production API/Web deployment boundary and release identity;
3. rehearse backup/restore against the production-shaped database path;
4. qualify a physical iOS device/TestFlight build;
5. run a small owner pilot focused on whether Today, Adventure, Skillcraft, Community, and Packs feel useful rather than chore-like;
6. use observed retention, confusion, safety stops, social participation, and repeat use to choose the next product tranche.

If cooperative Pack play emerges as a real pull from users, receipt-backed Expeditions become a strong next candidate. Until then, launch reality is more valuable than adding another game mechanic.
