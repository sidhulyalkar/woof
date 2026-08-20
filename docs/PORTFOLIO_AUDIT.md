# Woof Portfolio Hardening Audit

This document records the engineering and product audit used to turn Woof from an ambitious prototype repository into a clearer portfolio case study.

It is intentionally candid. A strong portfolio project should make it easy to distinguish **implemented behavior**, **experimental research**, and **future production work**.

## Executive assessment

Woof started with unusually broad ambition: web and native clients, social features, events, activity tracking, compatibility modeling, graph ML experiments, automation, storage, push, analytics and deployment infrastructure.

That breadth was also the main portfolio risk. Several layers looked more complete in documentation than they were in the canonical product path. The repository contained duplicate historical applications, conflicting product names, raw Prisma mutation contracts, mock/fallback personas, random compatibility scores, client/server drift, and CI configuration that described checks more strongly than the repository could actually reproduce.

The hardening pass therefore optimized for **coherence, truthfulness and system boundaries**, not feature count.

## Original risks and what changed

| Area | Original issue | Hardening decision |
| --- | --- | --- |
| Product identity | Woof and PetPath appeared in the same active experience | Standardized the canonical product on **Woof** across web metadata, login, PWA manifest, service worker and mobile entry screens |
| Repository topology | Multiple root-level app implementations competed with the monorepo | Preserved early FastAPI/mock frontend work under `legacy/`; established `apps/web`, `apps/mobile`, `apps/api` as canonical |
| Documentation | Many “FINAL”, “COMPLETE”, session and readiness reports dominated the root | Preserved them under `docs/archive/`; replaced the front door with architecture, product, ML and security documentation |
| Compatibility | Canonical NestJS path generated random scores | Added deterministic, explainable baseline with provenance, confidence, factors and reasons |
| Advanced ML | Sophisticated Python models could be read as already product-proven | Added E0–E5 evidence ladder and explicit promotion gates in `docs/ML_SYSTEM.md` |
| Discovery | Web expected a different payload than the API returned | Added a normalized compatibility contract and removed mock discovery data |
| Authentication state | Web maintained two independent session stores | Made the persisted auth profile the canonical client state and refreshes it from `/auth/me` |
| Pet ownership | Pet creation accepted raw Prisma input and did not bind ownership to JWT | Added app DTOs, owner binding and owner-only update/delete checks |
| Onboarding | Collected data that was never persisted; quiz answers disappeared | Reduced collection to useful fields, uploads optional media after account creation, persists matching preferences to `QuizResponse` |
| Privacy | Registration asked for location-like context before feature need | Removed unnecessary location/age collection from account creation and documented contextual permission boundaries |
| Profiles | Fallback persona and fake metrics could appear when the API failed | Profile now fails explicitly, uses real account/pet/gamification data, and has a real edit endpoint |
| Social writes | Raw Prisma input trusted actor IDs in bodies | Added actor-bound DTO-driven post, like and comment mutations with ownership checks |
| Gamification | Likes could mint points | Removed points for likes because the incentive is trivial to farm and misaligned with IRL outcomes |
| API exposure | ORM projections could expose more fields than intended | Added explicit public/member versus self profile projections; removed owner email/device internals from pet responses |
| Secrets | JWT code had fallback secret; env example contained key-like VAPID values | Added environment validation, removed fallback JWT secrets, removed key material from examples |
| Optional integrations | Missing storage/push credentials created ambiguous failures | Push and media now degrade explicitly without preventing core application startup |
| Mobile auth | Client expected camelCase tokens, refresh flow and routes the API did not implement | Aligned mobile with `/api/v1`, `access_token`, `/auth/me`, and local token clearing on 401 |
| Mobile permissions | Native config requested broader storage/location permissions than necessary | Reduced to camera plus coarse/fine when-in-use location and contextual explanations |
| Feed | Web treated a paginated API envelope as a ready UI array | Added client adapter and viewer-like state; removed unsupported feed controls |
| PWA | Service worker still carried old product identity and broad caching behavior | Renamed cache/notifications to Woof and prevents API traffic from entering static cache |
| CI | Expected absent lockfile; formatter/lint behavior did not match check semantics | Added explicit format/lint/type/test/build gates and documented the lockfile gap rather than hiding it |

## Architectural principles enforced by the hardening pass

### 1. Authentication establishes actor identity

Mutation endpoints should derive the acting user from the validated JWT, never from a client-supplied `userId` or `ownerId`.

This is now true for the pet and social mutation paths touched by the hardening pass.

### 2. ORM models are not API contracts

Prisma types are useful inside the data layer but are too permissive and too close to persistence to serve as public mutation DTOs. Application DTOs now constrain pet, social, quiz and profile writes.

### 3. Client adapters absorb transport details

The API can return pagination envelopes and rich relational objects while presentation components operate on focused UI models. Compatibility and social feed adapters make this boundary explicit.

### 4. Optional infrastructure fails locally

A missing ML, push or object-storage dependency should not make account, profile or relational state unusable.

### 5. ML earns promotion with evidence

A deterministic baseline is not glamorous, but it is essential. It gives the project a reproducible control, a degradation path and a way to ask whether a GNN is actually worth its operational complexity.

### 6. Location is a sensitive capability, not a profile decoration

Home coordinates, routes and routine schedules deserve separate permission, retention and display decisions. They should not be collected simply because a map feature may exist later.

### 7. The UI should not advertise dead affordances

A polished interface with buttons that only `console.log`, routes that do not exist, or controls backed by mock state is less credible than a smaller real interface. The hardening pass removes or connects those affordances where they intersect the portfolio-critical flows.

## Portfolio evidence matrix

| Capability | Current evidence | Maturity |
| --- | --- | --- |
| Web product shell | Next.js app with unified Woof design tokens, mobile-first navigation, real auth/profile/feed/discovery paths | Implemented |
| Native entry/auth | Expo app with aligned auth protocol, secure token storage and reduced permissions | Implemented |
| Account authentication | NestJS JWT auth with fail-fast secret configuration | Implemented |
| Pet ownership | JWT-bound creation and owner-only mutations | Implemented |
| Social actor authorization | Actor-bound posts/likes/comments and owner-only edits/deletes | Implemented |
| Compatibility baseline | Deterministic scoring with provenance, confidence, factors and explanation | Implemented |
| Advanced graph/temporal ML | Python training/inference research artifacts | Experimental |
| Preference learning signals | Authenticated onboarding preference persistence | Implemented |
| Realtime messaging | Socket.IO infrastructure exists | Implemented, further scale validation required |
| Object storage | S3/R2-style integration with explicit unconfigured degradation | Integrated when configured |
| Web Push | Web Push integration with explicit unconfigured degradation | Integrated when configured |
| CI | Workflow now describes executable quality gates | Implemented configuration; green run still needs evidence |
| Reproducible JS dependency graph | No committed `pnpm-lock.yaml` at time of audit | **Gap** |
| Public synthetic demo | Not established by this audit | **Gap** |
| Production moderation / abuse operations | Architecture documented, complete operational system not demonstrated | **Gap** |
| Online ML outcome lift | No controlled real-user evidence yet | **Gap / E5 target** |

## Full-scale questions a reviewer should ask

### What happens when local network density is low?

Recommendation quality cannot create candidates that do not exist. A real launch should concentrate geographically and measure candidate-set coverage independently from ranking quality.

### How do you prevent pairwise graph explosion?

Do not materialize every possible pair. Persist observed/meaningful relationship edges and perform constrained candidate generation before expensive ranking.

### How do you protect home and route information?

Separate precise private location, coarse candidate-discovery location and public meetup location. Public surfaces should never receive raw home coordinates by default.

### How do you know a compatibility score means anything?

Use leakage-resistant holdouts, calibration, provenance, deterministic baselines and eventually controlled online experiments tied to attended/repeated meetup outcomes.

### What happens when the model service fails?

The core API should time out quickly and use the deterministic baseline. A model outage is a recommendation-quality degradation, not an application outage.

### How does realtime scale?

Persist messages independently of socket delivery, use shared pub/sub for multiple Socket.IO nodes, make reconnect/catch-up idempotent and re-authorize room joins.

### How do you stop engagement incentives from corrupting the product?

Avoid rewarding trivially farmable clicks. Prefer successful activity, meetup, relationship and community outcomes while keeping safety metrics as hard guardrails.

## Remaining gates before I would call the project production-ready

1. Commit and validate a `pnpm-lock.yaml` and move CI back to frozen installs.
2. Observe a clean CI run for format, lint, TypeScript, unit, API, browser and production-build gates.
3. Add focused authorization tests for every sensitive resource family, not only pets/social.
4. Complete account deletion/export and data-retention workflows.
5. Add real blocking/reporting/moderation operations for IRL meetups.
6. Add geospatial privacy tests and fuzzing/coarsening rules.
7. Integrate the strongest ML candidate behind the same compatibility contract and collect latency/fallback telemetry.
8. Benchmark models against the deterministic baseline on time-, owner- and pet-disjoint splits.
9. Add automated accessibility and visual-regression coverage for core web flows.
10. Deploy a synthetic-data public demo without sensitive real-world route/location data.
11. Run backup/restore and failure drills for the hosted data plane.
12. Run an online experiment only after trustworthy outcome labels exist.

## Suggested portfolio walkthrough

A concise demo can tell the whole systems story in roughly this order:

1. **Login:** explain the product thesis and data-minimizing account boundary.
2. **Home:** show intent-first actions rather than an engagement-first feed.
3. **Discover:** open compatibility details and point out baseline provenance, confidence and explanation.
4. **Profile:** show the pet model and separately persisted matching preferences.
5. **Architecture:** explain why model inference is replaceable and PostgreSQL owns product truth.
6. **ML system:** walk through E0–E5 and why training a GNN is not the same as proving product lift.
7. **Scale/safety:** discuss graph candidate generation, geospatial privacy, realtime fanout and meetup moderation.
8. **Engineering discipline:** show CI, security policy, archived prototypes and the explicit remaining gaps.

The most defensible portfolio message is not “I built every feature.” It is:

> **I can take a broad, research-heavy product idea and turn it into a system with clear contracts, failure behavior, privacy boundaries, evidence standards and a path from prototype to production.**
