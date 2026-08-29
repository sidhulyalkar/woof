# Woof Repository Security Baseline

Woof's repository boundary should follow the same principle as dogOS runtime authority: no single presentation or convenience layer is allowed to silently become release authority.

## Current repository finding

As of August 29, 2026, GitHub reports the `main` branch as unprotected and the repository has no active repository rulesets. This is an administrative control gap, not an application-code defect.

The connected automation used for repository maintenance does not have access to create or edit branch-protection rules. The owner must configure the repository rule in GitHub after the code-level security baseline lands.

## Required `main` ruleset

Create an active repository ruleset targeting the default branch and configure the following:

1. Require a pull request before merging.
2. Require the branch to be up to date before merge, or use an equivalent merge-queue policy that still proves the exact candidate head.
3. Block force pushes.
4. Block branch deletion.
5. Require conversation resolution.
6. Require the stable, always-running release checks listed below.
7. Keep bypass authority minimal and reserved for incident recovery. Normal feature work must not use a bypass.

### Required stable checks

Use the exact check names GitHub presents after the workflows have run successfully:

- `Static quality gates`
- `Backend tests`
- `Frontend tests`
- `Production builds`
- `Maintained action runtime + project toolchain qualification`
- `Supply-chain + secret hygiene`

After CodeQL has produced its first successful analysis on `main`, also require the JavaScript/TypeScript and Python CodeQL analysis checks if the repository ruleset UI exposes them as stable required checks.

Do **not** globally require every path-filtered dogOS feature workflow. GitHub can leave a required path-filtered check pending when that workflow legitimately does not trigger. Feature-specific authority lanes remain required release evidence for pull requests that touch their domain and high-risk merges continue to use expected-head guards.

### Review count for a single-maintainer repository

Do not configure a mandatory approval count that the sole maintainer cannot satisfy on their own pull request. Until a second trusted maintainer exists, require pull-request review workflow, conversation resolution, and required checks with zero mandatory external approvals. Once a second trusted maintainer is active, raise the rule to at least one required approval for security-sensitive changes.

## Supply-chain policy

The committed baseline provides:

- controlled weekly Dependabot updates for the pnpm/npm dependency graph;
- controlled weekly Dependabot updates for GitHub Actions;
- CodeQL analysis for JavaScript/TypeScript and Python;
- a high-confidence committed-secret scan that never prints matching credential values;
- a production dependency audit that blocks high and critical advisories pending triage;
- a reproducible production dependency inventory artifact;
- deterministic GitHub Action runtime/version qualification.

## Dependency vulnerability triage

The automated audit is intentionally a first-line release gate, not the final interpretation of risk.

- Critical reachable production vulnerabilities block release.
- High reachable production vulnerabilities block release unless risk acceptance is explicit, time-bounded, owned, and documented.
- Development-only or non-reachable findings are tracked rather than silently ignored.
- Do not use `--force` upgrades solely to make scanners green.
- Any exception should identify the advisory, affected package, production reachability, compensating controls, owner, and expiry.

If the audit produces a legitimate false positive or non-reachable high advisory, add a narrowly scoped documented exception mechanism rather than disabling the audit globally.

## Secret hygiene

The committed scanner checks tracked text files for high-confidence credential classes such as private keys, GitHub tokens, AWS access keys, Slack webhooks, live Stripe keys, OpenAI API keys, Fly tokens, and npm tokens.

A scanner pass does not prove that no secret exists. GitHub secret scanning and push protection should also be enabled in repository settings when available. If a real credential is ever committed, rotate it even if the commit is later removed.

## Release authority checklist

Before calling a SHA public-beta ready:

- stable required checks are green on the exact head;
- feature-specific authority lanes relevant to the change are green;
- CodeQL has no unresolved release-blocking finding;
- production dependency audit is green or every blocking exception is documented and unexpired;
- committed-secret scan is green;
- production deployment credentials are configured through repository/environment secrets, never source;
- deployment readiness, rollback, and backup/restore evidence is current.

A qualified application SHA must not be able to reach `main` or production by bypassing the explicit authority checks that dogOS itself expects for user and pet state.
