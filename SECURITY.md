# Security Policy

Woof treats authorization, privacy boundaries, evidence provenance, and release authority as security properties.

## Reporting a vulnerability

Please do not publish exploit details, credentials, private user data, or proof-of-concept payloads in a public issue.

If GitHub private vulnerability reporting is available for this repository, use **Security → Report a vulnerability**. If it is not available, contact the repository owner through their GitHub profile with a minimal non-sensitive note asking for a private reporting channel. Do not include vulnerability details until a private channel is established.

A useful initial report includes:

- the affected Woof component and commit or release;
- the security boundary you believe can be crossed;
- reproducible steps that avoid accessing data you do not own;
- expected versus observed behavior;
- whether exploitation requires authentication, a particular role, or race timing;
- any safe mitigation you have already identified.

## Supported code

Security work targets the current `main` release line and canonical applications under `apps/`, `packages/`, `ml/`, `infra/`, and `n8n/`. Historical material under `legacy/` and `docs/archive/` is preserved for provenance and is not an active deployment boundary.

Until Woof begins tagged public releases, older development branches are not supported security boundaries.

## Release vulnerability policy

Woof does not treat a scanner's raw advisory count as release truth. Findings are triaged by production reachability, severity, authority impact, and available mitigation.

- **Critical reachable production vulnerability:** blocks release.
- **High reachable production vulnerability:** blocks release unless there is a documented, time-bounded risk acceptance with an owner, rationale, compensating control, and expiry date.
- **Moderate/low or development-only finding:** must be triaged and tracked; it does not become invisible merely because it is non-blocking.
- **Credential exposure:** blocks release and requires credential rotation. Deleting the committed value alone is not sufficient.
- **Security control regression:** a failing authorization, session, migration, secret-hygiene, or release-authority contract blocks release even when dependency scanners are otherwise green.

Do not use forced dependency upgrades solely to silence audit output. Preserve reproducibility and re-run the full relevant qualification matrix after security-sensitive changes.

### Dependency risk acceptance

High or critical dependency findings are not suppressed through a global ignore list. Any temporary acceptance must live in `.github/security-audit-exceptions.json` and pass `.github/scripts/assert-pnpm-audit-policy.py` against fresh `pnpm audit --prod --json` evidence.

The current exception mechanism is deliberately narrow:

- one exact GHSA identifier per entry;
- exact package and severity matching;
- a named owner and written rationale;
- a maximum 45-day acceptance horizon;
- no wildcard advisories, packages, or dependency paths;
- all affected dependency paths must remain inside the explicitly accepted boundary;
- stale exceptions fail instead of lingering after the dependency graph changes.

The only supported exception class today is `mobile-build-tool-only`, scoped to dependency paths beginning with `apps/mobile >` and carrying the required Expo/Metro path markers. A finding that moves into Web, API, shared runtime code, or another dependency path becomes release-blocking immediately even if its GHSA identifier was previously accepted.

## Security boundaries that matter most

The highest-risk areas in Woof are:

1. authorization across user-owned pets, temporary caregivers, households, conversations, activities, and meetup data;
2. precise home, route, and live-location exposure;
3. account and persisted-session authority;
4. file upload validation and object access;
5. realtime room authorization and revocation;
6. abuse, blocking, and reporting for real-world coordination;
7. service-to-service trust around ML, connectors, devices, and automation integrations;
8. software-supply-chain and deployment authority.

The following separations are intentionally fail-closed:

- realtime authentication does not replace persisted session authority;
- caregiver grants do not become household membership, medical authority, reward authority, or owner evidence;
- cached presentation state does not override current server authority;
- model output does not become release authority;
- social score does not become recommendation or model truth;
- missing evidence remains missing rather than being synthesized into certainty.

Security fixes should preserve those separations rather than bypassing them for convenience.

## Production readiness

Production-oriented controls such as authentication, validation, security headers, rate limiting, observability, migration contracts, and deployment gates are necessary but not sufficient. Public-beta readiness also requires repository protection, dependency/static analysis, secret hygiene, production credential authority, backup/restore rehearsal, rollback proof, abuse tooling, and deployment-specific verification.
