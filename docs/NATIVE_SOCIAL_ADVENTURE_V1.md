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

### Local Packs

Packs are account-level social communities and are available to Guardian and Companion modes. Joining a Pack never grants pet authority.

A Pack uses a coarse, user-chosen `regionKey`, such as `south-bay-ca`. Native Woof does not request device location to create or rank a Pack, and it does not derive locality from a home address, route endpoint, GPS trace, or meetup location.

Local standings are fail-closed. The client only renders entries when the server returns `cohortReady: true`. If the server cohort is not ready, native Woof displays the server's privacy message and `memberCount / minimumCohort`. It never calculates whether a cohort should be considered safe itself.

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

If authority cannot be loaded, native Woof shows unknown/unavailable state. It must not substitute cached-looking zeroes, guessed ranks, fabricated privacy state, or inferred location.

## Companion mode

Companion users can participate in pet-independent Human Skill, Community, and Packs without inventing a dog. These account-level social surfaces do not open Today, Compass, Story, Daily Signals, or any other pet-authorized surface.

A Companion user's Social Adventure score is whatever the server can legitimately derive from that account. The client does not manufacture Adventure evidence to make the league look fuller.

## Why Expeditions are not in this tranche

The product direction still includes cooperative seasonal Expeditions and Pack goals, but there is not yet a canonical Expedition API or receipt model in the server.

Native v1 therefore stops at Feed + global league + Packs + privacy-safe Pack standings. Shipping a decorative client-side Expedition counter would create exactly the kind of unowned authority dogOS is designed to avoid.

A later Expedition tranche should add server-owned templates, participation receipts, category caps, duplicate protection, recipient/pet eligibility where relevant, and cooperative progress before native presentation is built.

## Qualification contract

`Native Social Adventure CI` freezes these boundaries by checking that:

- native Community no longer imports the legacy social API;
- feed, score, preference, rank, reaction, and Pack data come from Social Adventure endpoints;
- only the five canonical semantic reactions are exposed;
- global visibility changes through one explicit user action and is never auto-enabled;
- Pack standings are rendered only behind server `cohortReady`;
- native Packs contain no device-geolocation implementation;
- client code does not sort or derive league ranks;
- native social types omit pet ID and legacy like/comment counters;
- the server Social Adventure score policy tests are rerun;
- the full native client still type-checks and lints with zero warnings.

## Next game layer

After this convergence qualifies, the next high-value social-game release is cooperative **Expeditions**.

The right version is not “do 10 walks.” It is a bounded seasonal map where a Pack collectively discovers varied safe experiences and human skills. A contribution should be based on authoritative eligible receipts, capped for repetition, and able to celebrate recovery, learning, and opting out. The event should create a shared story, not an exercise quota.
