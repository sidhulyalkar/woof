# dogOS Architecture

## Product thesis

dogOS is the household operating layer for life with dogs. The product should reduce care work, create more opportunities for a good relationship, or ideally do both.

North star: **I take both dogs for a walk. I tap once. dogOS understands everything else.**

## Foundation boundary

The household is the coordination and authorization boundary:

```text
Household
├── humans
├── pets
└── shared real-world activities
    ├── human participants
    └── pet participants
```

One physical event is one `Activity`. A walk involving multiple dogs is represented by one activity with multiple `ActivityPetParticipant` rows, never duplicated activity records.

## Activity write contract

The canonical client request is:

```ts
interface CreateActivityRequest {
  petIds?: string[];
  /** @deprecated Prefer petIds, even for one dog. */
  petId?: string;
  householdId?: string;
  startedAt?: string;
  endedAt?: string;
  type: string;
  route?: Record<string, unknown>;
  humanMetrics?: Record<string, unknown>;
  petMetrics?: Record<string, unknown>;
  jointMetrics?: Record<string, unknown>;
}
```

`petIds` is canonical. `petId` remains accepted for old clients. When either or both are present, the server de-duplicates the set and resolves a single household containing the selected pets. A request cannot combine pets from unrelated households.

The currently shipped mobile form contract is normalized at the mobile API boundary before it reaches Nest validation. This preserves strict server-side whitelisting while allowing an incremental client rollout.

## Reward semantics

Adventure rewards remain server-authoritative:

```text
Activity -> CareEvent -> RewardPolicy -> RewardLedger
```

A shared activity may create one trusted care event per participating pet. The historical primary-pet dedupe key is retained so migrating a single-pet record cannot create a second reward. Additional pets receive stable pet-specific dedupe keys.

## Privacy and safety invariants

- Household and pet access fail closed.
- Operational logs must not contain raw user, household, pet, token, location, or health identifiers unless a separately reviewed operational requirement explicitly demands it.
- Nearby discovery must use an explicit privacy-preserving backend and consent model; clients must not emulate it by downloading broad owner/location data.
- Tracker or behavioral deviations are signals, not diagnoses.
- Medical or emergency flows are never gamified.
- Connector access is least-privilege and revocable.

## Database and rollout invariants

- New dogOS schema changes are additive-only.
- Existing `Pet.ownerId` and `Activity.petId` remain compatibility shims during migration.
- Prisma `String @default(uuid())` identifiers in this repository are stored as PostgreSQL `TEXT` unless explicitly annotated otherwise.
- Every dogOS layer begins dark behind a feature flag where it changes user-visible behavior.
- Rollback means disable the feature and/or revert application code; additive history is retained rather than destructively down-migrated.

## Layering

```text
dogOS
├── Today
│   ├── Adventure
│   ├── Autopilot
│   └── Our Story
├── Household Wellbeing Graph
│   ├── Humans
│   ├── Pets
│   ├── Activities
│   ├── Care
│   └── Costs
└── Connectors
    ├── calendars and weather
    ├── wearables and care records
    ├── providers
    └── retail/services where sanctioned integrations exist
```

Higher layers compose lower-layer truth instead of creating parallel copies. Autopilot derives signals from household activities and care. Our Story composes activities, care events, Adventure events, and private media. Connectors normalize sanctioned external data into dogOS contracts. Concierge orchestrates those layers rather than owning a second recommendation database.

## Phase 4A qualification contract

The dogOS Foundation CI lane owns:

1. frozen dependency installation;
2. Prisma validation, canonical formatting, generation, full migration deploy, destructive-operation audit, and zero database/datamodel drift;
3. Prettier on dogOS and touched mobile compatibility surfaces;
4. zero-warning API lint on household/activity/pet integration surfaces;
5. API and full mobile TypeScript checks;
6. household authorization contracts;
7. multi-pet activity/reward contracts; and
8. production API build.

The parent Adventure qualification must remain green on the same candidate head. PR #8 stays draft while its parent release branches remain unlanded; after the parent stack reaches `main`, dogOS must be restacked and qualified again.
