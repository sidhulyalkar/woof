# First Adventure release contract

First Adventure is the relationship-first entry into Woof. It is intentionally not a profile-completion funnel.

## Product promise

The first useful loop is:

`create the pair -> learn only what helps now -> do something together -> observe -> adapt`

The app should earn screen time by helping the owner choose, notice, handle, adapt, or remember. It should not make questionnaire completion the price of reaching a real activity.

## Setup boundary

Canonical onboarding has three phases:

1. **Human account**: the minimum durable account fields needed to authenticate.
2. **Pet identity**: durable basics needed to create the authorized household/pet pair. Breed is optional supporting context. Temperament is not demanded as an instant personality label.
3. **First Adventure**: three small, optional personalization moments that can help break ties between otherwise safe suggestions.

The account and pet are durable before phase 3. Sparse personalization never blocks Today.

## First Adventure questions

The initial moments cover:

- up to three current owner goals,
- realistic time and effort,
- one high-value social-comfort signal.

Every field may be skipped. `Not sure` is explicit uncertainty, not dislike. These answers are Adaptive Profile evidence only and cannot award Bond XP, alter the RewardLedger, create streak authority, or manufacture mastery.

## Retry and trust rules

- Canonical email registration carries a client-generated UUID `registrationKey`.
- The server canonicalizes the new email/handle and derives a deterministic user UUID from `(canonical email, registrationKey)`.
- An exact registration retry must prove the deterministic account identity, original account fields, and original password hash before it can recover. A successful recovery issues a fresh server-owned session; it never resurrects a lost token.
- Replaying a registration key with divergent fields, or presenting the same email under a different key, fails closed as a conflict.
- Older registration clients that do not send a replay key retain ordinary one-shot uniqueness behavior.
- Minimal onboarding pet creation may carry a replay-safe `creationKey`.
- The API derives one deterministic pet identity from `(owner, creationKey)`.
- Exact pet retries converge on the existing pet.
- Replaying the pet key with different identity fields fails closed.
- Media and mutable JSON are not accepted in replay-safe pet creation. Pet photos attach only after the pet exists and are best effort.
- A browser-stored `(household, pet)` pair is only a recovery hint. The server must authorize it again before First Adventure resumes.
- Editing pet details from First Adventure updates the durable pet; it does not create another pet.
- Adaptive Profile write failures are non-blocking. The pair still enters Today and Woof can learn optional context later.

## Relationship-first interaction rules

First Adventure should feel calm rather than evaluative:

- no profile-completeness score,
- no guilt copy for skipping,
- no fake precision,
- no reward bait for supplying data,
- touch targets sized for mobile use,
- plain-language explanations for why a question matters,
- owner and pet choice remain visible in social-comfort language.

## Qualification

The release is qualified with:

- a dedicated First Adventure CI lane,
- a dedicated registration-replay CI lane,
- zero-warning targeted Web and API lint,
- Web and API TypeScript,
- focused First Adventure, Adaptive Profile, authentication replay, pet transport, and replay-safe service contracts,
- desktop and mobile Playwright onboarding journeys, including a simulated lost registration response,
- root browser/accessibility/visual contracts,
- API and Web production builds.

## Transaction boundary

Account and pet creation now use separate replay identities because they are separate durable transactions. The browser keeps each key only as long as that transaction may need recovery. A successful registration response clears the account replay key; completing onboarding clears any remaining recovery hints. This keeps retries recoverable without turning browser storage into account authority.
