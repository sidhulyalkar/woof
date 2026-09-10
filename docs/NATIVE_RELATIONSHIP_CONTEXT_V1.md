# Native Relationship Context v1

## Goal

Make Woof's dog-specific surfaces truthful and calm for multi-dog households.

Today and Compass now share one explicit relationship selection instead of silently relying on the server's default pet resolution. The selection is presentation state only. Every dashboard request still sends the selected pet ID to canonical server authorization.

## Scope model

Woof should make the scope of a surface obvious:

- **Today**: one selected dog-human relationship.
- **Compass**: the same selected dog-human relationship.
- **Story**: remains household/user history by default and may later add an explicit pet filter.
- **Skillcraft, Expedition, Field Journal, Community**: human-side surfaces, not a selected dog's score.

This release changes only Today and Compass.

## Household authority

The mobile relationship store reads `GET /households/me`. That endpoint already returns only active viewer memberships and active household-pet links.

The chosen pet ID may be stored locally with SecureStore so the app can reopen to the same relationship. That stored value grants nothing. On load, it is intersected with the fresh household snapshot. If it is no longer authorized, Woof discards it and chooses from the current authorized set.

The storage key is namespaced by signed-in user ID so a logout/login on the same device cannot reuse another account's pet preference.

## Multi-household behavior

A pet can appear through more than one household relationship. The mobile presentation deduplicates by canonical pet ID and retains household names only as non-authoritative display context.

The server still performs authorization on every requested `petId`.

## Cross-pet stale-state boundary

Today and Compass only render a loaded dashboard when `dashboard.pet.id === selectedPetId`.

When the user changes dogs, local quest/result state is cleared and the previous dog's dashboard is removed before the new request. A stale response therefore cannot be presented as belonging to the newly selected relationship.

## Coverage scale repair

Canonical `CareSummary` pathway coverage is a percentage from 0 to 100:

```text
coverage = min(100, recentDays * 25)
```

Native Compass previously treated the same value as a 0–1 fraction, clamping any value above 1 to 1 and multiplying by 100 again. One recent day (`25`) could therefore display as `100%`.

Compass now clamps directly to 0–100 and uses that percentage unchanged for the visual width and label.

## Accessibility

Relationship choice chips use a 44pt minimum target and expose selected state through accessibility metadata. The single-pet presentation is non-interactive because there is no choice to make.

Several small Today actions touched in this tranche were also raised to a 44pt minimum target.

## Deliberately deferred

This release does not:

- make Story inherit the selected Today/Compass dog;
- create a new pet-authority endpoint;
- use local storage as authorization;
- combine histories across dogs;
- alter Adventure reward policy;
- change Social Adventure, Expedition, Skillcraft, or Journal scoring;
- remove the existing server fallback for older clients that omit `petId`.

A later calm-experience tranche can add an explicit `All dogs / <dog>` Story filter and continue reducing visible accounting without changing canonical evidence.
