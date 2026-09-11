# Native Scope Clarity v1

## Purpose

Native Woof now has several useful surfaces that intentionally operate at different scopes. This tranche makes those boundaries visible in the product and executable in CI so convenience cannot quietly become authority.

The design goal is not to add more warnings. It is to make the default behavior truthful enough that the user rarely needs one.

## Three scope classes

### Today + Compass

Today + Compass are relationship-scoped dog surfaces.

- Their selected dog must come from fresh authenticated household authority.
- The selected pet is only a presentation preference. Local selection never grants authority.
- The client sends the selected `petId` explicitly to the canonical Adventure read.
- A response for another pet is hidden rather than relabeled.
- Changing dogs clears transient cross-dog presentation state instead of carrying it forward.

This is the narrowest scope in the native shell because the user is asking about one concrete dog-human relationship.

### Story

Story has a different truthful default.

The canonical all-dogs Story is the signed-in human's authorized household history returned by `GET /story` without a client-selected pet filter. It is not the currently selected Today + Compass relationship and it does not inherit that preference.

A user may narrow Story to one dog. Filter options are discovered from authenticated `GET /households/me`, and the selected `petId` is still re-authorized by the Story service. A local chip therefore narrows a server-authorized view; it does not create access.

Network behavior is intentionally asymmetric:

- failure to discover optional dog filters does not erase or block the canonical all-dogs Story;
- an old Story response cannot replace a newer selected scope;
- a response is rendered only for the scope that requested it;
- failed scoped reads leave that scope unavailable rather than borrowing another dog's Story.

Story remains a read model over canonical source history. This tranche does not add Story mutation authority.

### Skillcraft + Expedition + Field Journal + Community

Skillcraft + Expedition + Field Journal + Community are human-side surfaces. They are useful before a person has a dog and do not require synthetic pet scope.

Companion mode can therefore enter these spaces directly. Presentation mode does not manufacture a pet relationship or unlock pet-specific Today, Compass, or Story.

Expedition has two live cooperative scopes:

- Global, read from canonical Global Expedition authority;
- a joined Pack, read only after the server catalog confirms `joined: true` and the returned Pack identity matches the requested Pack.

The Field Journal is personal Global history over immutable Expedition receipts. It is not Pack history and it is not reconstructed from the live world, Social Adventure score, Story, route history, or client arithmetic.

## Calm cooperative presentation

The server retains bounded Expedition totals for calibration, but native presentation deliberately does not advertise repeatable personal volume.

Personal Expedition repetition is binary presence. A person either has a server-issued mark for a landmark or does not. Repeating the same qualifying category does not make a larger personal mark in the world or Field Journal.

Communal contributor counts are descriptive context, not a target. There is no client completion percentage, fallback target, progress bar, unlock threshold, rarity ladder, or finish-line arithmetic while authority remains `CALIBRATING`.

Every landmark exists from the beginning. The world can become visually richer without turning a dog's health, mileage, duration, missed days, popularity, or repetition volume into game pressure.

## Independent failure domains

Native reads should preserve the smallest truthful unit of successful authority.

- Story filter discovery can fail while the all-dogs Story remains usable.
- Global Expedition, Pack catalog, and a selected Pack projection are request-bound so stale responses cannot win after a newer refresh or scope change.
- A Pack response must identify the Pack that was requested before display.
- Journal failure does not poison the live world.
- When a previously verified Journal exists, a refresh failure may continue showing those verified pages with stale-state copy.
- When no verified Journal exists, history remains blank rather than being guessed from current Expedition data.

The UI may preserve previously loaded server-confirmed data after a transient read failure, but it must describe that state honestly and never synthesize missing authority.

## Petless Companion experience

A person in Companion mode can belong in Woof before they have a dog. The native home therefore leads with human-side places to learn, participate, and meet people rather than presenting an empty or broken pet dashboard.

If their situation changes, switching presentation mode still creates no pet access. Canonical onboarding must resolve a real relationship before dog-specific surfaces open.

## Accessibility and interaction

Scope selectors are real buttons with selected accessibility state, not decorative labels. Touched native scope controls keep at least a 44pt interactive height so the clearer authority model is also practical to use.

## Permanent sentry

`assert-native-scope-clarity.py` fails closed if the product drifts across these boundaries. It is intentionally cross-cutting and runs alongside the more specific First Adventure, Native Expedition World, and Expedition Field Journal contracts.

The dedicated CI lane also compiles all four sentries, installs from the committed lockfile, rejects formatting drift, type-checks the complete mobile client, and lints mobile with zero warnings.

## Non-claims

This tranche does not claim:

- that a local selection grants authorization;
- that Story's all-dogs view means lifetime-complete history;
- that Companion mode creates or implies a pet relationship;
- that Expedition contributor counts are calibrated goals;
- that Field Journal is a completion system or Story persistence;
- offline authority;
- physical-device or TestFlight qualification;
- production deployment or pilot evidence.

Those remain separate authorities and release gates.
