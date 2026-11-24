# Native Daily Signals vNext v1

## Purpose

This release makes Daily Signals feel like a quick observation, not a six-item daily form.

The canonical backend capture authority is unchanged:

- one answered dimension is enough;
- untouched dimensions are absent evidence;
- `UNSURE` is explicit uncertainty and does not project into the baseline;
- a different second payload for the same pet + household-local day still conflicts;
- notes remain private and do not become signal labels.

## Native interaction

The maintained iOS/React Native surface now asks:

> Anything different today?

The user sees six compact rows. Every untouched row says `Not reported`. Tapping one row exposes `Less / Usual / More / Not sure`, plus `Leave unreported`.

Only one row needs to be open at a time. There is no completion percentage and no implication that all six should be answered.

The save button reports the number of observations being written. A user with nothing meaningful to report can choose `Nothing to add today`, which simply leaves the screen. It does not create six synthetic `USUAL` observations.

The optional private note is collapsed by default and cannot be saved by itself.

## Relationship-context handoff

Today may pass the selected relationship pet as a preferred navigation context.

Daily Signals still loads canonical household memberships from the server. It preselects a context only when:

- there is exactly one authorized context for the preferred pet; or
- there is exactly one authorized pet/household context overall.

If a pet appears in multiple authorized household contexts, the screen requires an explicit choice. Array order is never household authority.

## Success receipt

After capture, the UI states how many observations were saved and for which dog.

If `UNSURE` was included, the receipt explicitly says it remains uncertainty rather than a baseline value. The receipt also states that Daily Signals are private context, not a health score.

## Accessibility

- signal choices are wrap-capable;
- interactive choices use at least 44-point minimum height;
- each compact row exposes its current value or `Not reported` to accessibility APIs;
- the note remains inside the existing keyboard-avoiding navigation wrapper;
- no control depends on color alone.

Repository qualification is not physical-device evidence. Largest Dynamic Type, VoiceOver traversal and real keyboard behavior remain part of the authenticated/device gates.

## Explicit corrections

Native now loads the effective structured state before permitting capture or editing.
`Edit observations` begins a correction using the current CareEvent as the expected
predecessor. The correction includes the complete intended structured state; an
empty state explicitly retracts all observations. Private notes are not editable
through this endpoint and are never copied into correction requests.

An uncertain save retains the same payload for retry while this screen remains
mounted. Conflicts require an explicit reload before another correction; reloading
replaces the draft. Responses for a previously selected dog cannot update the
current editor. Full offline persistence and recovery after process termination
remain future work.

Behavioral tests execute the maintained component using DOM primitives in
`apps/web/src/lib/native-daily-signals.spec.tsx`. They supplement source boundary
checks; they do not establish native layout or VoiceOver quality.
