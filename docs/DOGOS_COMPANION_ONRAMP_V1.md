# dogOS Companion Onramp v1

Companion Onramp makes Woof useful before, between, and beyond pet ownership without manufacturing a dog profile or weakening pet authorization.

## Product rule

**Account mode controls presentation. Pet relationships control pet authority.**

Those are separate systems.

A user may choose one of three presentation modes:

- `PET_GUARDIAN`
- `ANIMAL_ALLY`
- `FOSTER_CAREGIVER`

`AUTHORIZED_CAREGIVER` is intentionally not a global account mode in v1. Caregiver authority belongs to a bounded household/pet relationship, not an identity label that could accidentally grant access elsewhere.

## Landing state machine

The server resolves one of four landings:

- `NEEDS_MODE`: no selected mode and no authorized pet context;
- `NEEDS_PET_SETUP`: Pet Guardian selected, but no real pet relationship exists yet;
- `PET_TODAY`: the account has authorized pet context and may use the existing dog-human Today loop;
- `COMPANION_TODAY`: Animal Ally or Foster Caregiver experience.

A mode never creates a pet, household membership, or authorization edge.

Existing owners are migration-backfilled as `PET_GUARDIAN`. A user with shared household pet access but no persisted mode may temporarily reach `PET_TODAY` through `PET_AUTHORITY_COMPAT`; this preserves real relationship authority without falsely relabeling that account as the owner/guardian.

## Petless Today

Companion Today gives a useful loop without fake Bond XP or fictional dog state:

1. practice Human Skill games in Skillcraft Arcade;
2. participate in Community and privacy-bounded Packs;
3. use a private readiness reflection;
4. later switch to Pet Guardian if a real dog relationship begins.

Pet-only Compass, Story, Autopilot, and Coach navigation do not appear unless the server positively resolves `PET_TODAY`.

Navigation fails closed. If Companion state cannot be verified, the client exposes only pet-independent surfaces rather than guessing that a pet relationship exists.

## Readiness reflection

The readiness surface records six named dimensions:

- housing;
- household alignment;
- time capacity;
- financial plan;
- support plan;
- care plan.

Each dimension is one of:

- `NOT_SURE`;
- `WORKING_ON_IT`;
- `READY_TO_DISCUSS`.

There is deliberately **no readiness score**.

Woof does not convert the reflection into a percentage, rank, adoption recommendation, foster recommendation, financial assessment, housing assessment, or welfare clearance. The answers are private operational state and are not automatically copied into Social Adventure, pet state, recommendation evidence, or shelter workflows.

`READY_TO_DISCUSS` means only that the user feels prepared to discuss that dimension. It does not mean a shelter, rescue, foster program, landlord, veterinarian, trainer, or other responsible party has approved it.

## New-user onramp

The login surface now sends new users to `/onboarding/companion`.

The sequence is:

1. create the human account;
2. choose a starting role;
3. if Pet Guardian is selected, continue into the existing durable pet + First Adventure onboarding;
4. otherwise enter Companion Today immediately.

Registration keeps the replay-key semantics already used by First Adventure onboarding.

## Operational schema

Companion state lives in `dogos_companion`, separate from canonical Prisma models.

### `profiles`

Stores only:

- `user_id`;
- chosen presentation `mode`;
- timestamps.

It cascades on account deletion.

### `readiness_reflections`

Stores:

- `user_id`;
- a bounded JSON object of named readiness dimensions;
- timestamps.

It cascades on account deletion and contains no score column.

The migration backfills `PET_GUARDIAN` only from existing owned-pet rows. It intentionally does not turn shared household access into a permanent account identity.

## Real-world opportunities

v1 does not fabricate foster, volunteer, shelter, or adoption inventory.

Future opportunity surfaces must use partner-authorized sources and preserve the placement organization's eligibility and decision authority. Companion mode itself is not evidence that a user should receive an animal.

## Authority boundaries

Companion Onramp preserves the larger dogOS authority model:

- presentation mode != pet authorization;
- readiness reflection != adoption/foster assessment;
- Human Skill practice != professional credential;
- social score != recommendation truth;
- petless learning != fictional pet state;
- household caregiver access != global account ownership;
- shelter opportunity != placement authority.

## UI surfaces

- `/`: server-resolved router between Pet Today, Companion Today, mode choice, and pet setup;
- `/onboarding/companion`: human-first mode selection for new accounts;
- `/companion/readiness`: private, non-scored readiness reflection;
- `/arcade`: available without a pet;
- `/community`: available without a pet;
- bottom navigation: mode-aware and fail-closed for pet-only routes.

The previous guardian Today implementation is preserved intact behind the new router as `GuardianTodayPage`.

## Release invariants

Companion Onramp v1 must fail qualification if:

- Companion mode code creates a pet or relationship grant;
- Pet Guardian reaches pet Today without real authorized pet context;
- Animal Ally or Foster Caregiver requires a pet;
- pet-only navigation is exposed before `PET_TODAY` is positively resolved;
- a readiness score column or composite score is introduced;
- operational rows survive account deletion;
- the operational schema leaks into canonical Prisma state;
- API/web lint, type checks, tests, or production builds fail.

## Next releases

This is the foundation for the remaining #64 work. Follow-on releases should add:

1. explicit invitation/grant UX for authorized caregivers while keeping grants pet/household scoped;
2. partner-authorized shelter, foster, and volunteer opportunity connectors;
3. organization-owned eligibility forms instead of Woof inventing placement criteria;
4. transition receipts when a foster/adoption relationship begins, preserving human learning while starting dog-specific assumptions from zero;
5. richer petless learning paths reviewed with qualified trainers and shelter/foster professionals;
6. privacy controls for deleting or exporting Companion reflections.

The release succeeds if someone can use Woof honestly before owning a dog, while the full dogOS relationship loop still opens only when a real authorized pet relationship exists.
