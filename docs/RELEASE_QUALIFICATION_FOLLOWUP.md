# Release qualification follow-up: 2026-11-23

## Verified on PR #181

At commit `064f4bd139f1a59ff1bb623e7c09cbea631cc0f2`, GitHub CI applied the full
migration chain on PostgreSQL 15 with pgvector and verified Prisma schema parity.
Static quality gates, frontend unit/browser/accessibility contracts, native Daily
Signals, native accessibility source contracts, iOS artifact audit, and iOS
CocoaPods/Xcode qualification passed. These do not establish physical-device acceptance.

Backend execution passed 618 tests and failed one connector fixture. Its August
observation timestamp aged beyond the supported backfill window. This follow-up
anchors the suite clock, supplies the missing rejection metric mock, and verifies
that moving the clock beyond the window rejects the event before ingestion or
receipt creation. All 16 focused connector/contract tests pass locally.

## Dependency release blocker

The security job stopped at expired image-size exceptions. A fresh production
lockfile audit found 41 unique high/critical advisories across 19 packages (45
records because some advisories cover multiple installed versions). This is an
audit of dependency presence, not a claim that every advisory is exploitable in
Woof. The exceptions and policy remain unchanged; no release waiver is granted.

The registry-reported patch ranges below are evidence captured at audit time,
not a validated upgrade prescription. Re-audit and check upstream compatibility
before changing a version. In particular, image-size and Zod need major-version
compatibility work, and one node-forge advisory reports no patched range.

| Package              | Unique blocking advisories | Reported patched ranges                                              |
| -------------------- | -------------------------: | -------------------------------------------------------------------- |
| @sentry/nextjs       |                          1 | `>=10.75.2`                                                          |
| axios                |                          7 | `>=1.20.0`                                                           |
| body-parser          |                          1 | `>=1.20.9`                                                           |
| brace-expansion      |                          2 | `>=1.1.19`, `>=1.1.20`, `>=2.1.5`, `>=2.1.6`, `>=5.0.10`, `>=5.0.11` |
| compression          |                          1 | `>=1.8.2`                                                            |
| decode-uri-component |                          1 | `>=0.5.1`                                                            |
| engine.io            |                          1 | `>=6.6.10`                                                           |
| fast-uri             |                          2 | `>=3.1.7`                                                            |
| follow-redirects     |                          3 | `>=1.16.1`, `>=1.16.2`                                               |
| image-size           |                          2 | `>=2.0.3`                                                            |
| multer               |                          1 | `>=2.5.0`                                                            |
| next                 |                          2 | `>=15.5.28`                                                          |
| node-forge           |                          8 | `<0.0.0`, `>=1.4.1`                                                  |
| proxy-addr           |                          1 | `>=2.0.8`                                                            |
| sharp                |                          1 | `>=0.35.5`                                                           |
| shell-quote          |                          1 | `>=1.11.0`                                                           |
| source-map-js        |                          1 | `>=1.2.2`                                                            |
| undici               |                          4 | `>=6.28.1`, `>=6.29.1`                                               |
| zod                  |                          1 | `>=4.6.6`                                                            |

## Remediation sequence and acceptance

1. Update direct web/API packages within supported release lines first. Exercise
   authentication, middleware, uploads, realtime connections, image handling,
   and Sentry integration; run builds and browser contracts on the locked graph.
2. Refresh compatible transitive dependencies using parent upgrades or narrowly
   scoped overrides. Preserve different dependency major versions where their
   parents require them. Re-run the audit after each coherent group.
3. Qualify Expo/Metro separately. Inspect their image-size and certificate API
   usage before any override; build native bundles and repeat iOS qualification.
   Track `GHSA-86w9-cpqp-85rv` explicitly: the audit currently provides no patched
   range. An updated package alone is not evidence that this finding is resolved.
4. Migrate Zod consumers deliberately, including dependent tools. Do not impose
   a root-wide major-version override without checking their APIs and errors.
5. Remove exceptions only once the corresponding findings disappear. Any proposed
   remaining exception requires current dependency-path evidence and explicit
   scope/expiry under the existing policy. Do not merely roll expired dates forward.
6. Require a passing security policy, complete backend/API tests, production builds,
   and native qualification for the final commit. Keep PR #181 draft meanwhile.

## Next product slice after qualification

Skillcraft currently has four games with one fixed scenario each. Add reviewed
scenario variants with stable keys and retained historical scoring definitions.
Select the scenario on the server and return that exact persisted scenario;
`startHumanSkillAttempt` currently separately selects the first catalog entry by
challenge, which must change before multiple scenarios are enabled. Keep one room
per challenge in the catalog so variants do not duplicate room cards.

Then connect receipt feedback to an optional real-world practice/reflection flow.
Store simulation attempts separately from dog observations; a game score must
not raise a dog's coaching level. Preserve bounded weekly breadth credit and
no additional league credit for retries. Test scenario/receipt identity, old
attempt completion, retries, rejected/expired attempts, and native/web parity.
Content review and a real-device trial remain explicit acceptance gates.
