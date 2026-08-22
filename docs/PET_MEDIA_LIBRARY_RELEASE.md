# Pet Media Library release qualification

This document is the merge checkpoint for the integrated pet-intelligence / Health Lens / Behavior Vision / private Media Library branch used as the parent of the Adventure System.

## Release invariants

- Private pet media remains private by default and signed upload/download URLs are never persisted as durable asset data.
- Browser authentication and the Media Library use the same canonical `useAuthStore`; an authenticated owner's pet list must enable the library query without a second session store.
- The web Content Security Policy derives `connect-src` from the configured API **origin**, not a pathful API base. A value such as `https://api.example.com/api/v1` must allow descendant endpoints such as `/api/v1/auth/me` without broadening the policy beyond `https://api.example.com`.
- Media upload tests inspect Prisma arguments structurally so legitimate `bigint` byte counts remain supported while signed URLs and public visibility remain absent.
- Behavior Vision fails closed and its personal-profile tests must parse and execute before release.
- Health Lens emergency behavior remains outside game/reward mechanics.
- ML promoted-mode behavior retains artifact attestation and fallback guarantees.
- One-run branch-construction payloads, publishing workflows, finalize scripts, and formatting helpers are not part of the release candidate.

## Qualification defects resolved during integration

Fresh parent qualification exposed and fixed several defects that historical child-branch CI did not cover:

1. `@woof/ui` had no ESLint entry point. It now extends the repository's shared TypeScript base with strict reusable-library rules, while keeping `--max-warnings=0`.
2. `/library` read an obsolete session store while `AuthGuard` hydrated `useAuthStore`. The Media Library now consumes the canonical authenticated state.
3. `POST /auth/login` documented HTTP 200 but relied on Nest's default POST status 201. The controller now explicitly returns `HttpStatus.OK`, and backend e2e qualification passes that contract.
4. Cross-origin Playwright API mocks did not model CORS preflight behavior. The focused Library harness now handles OPTIONS and returns explicit CORS headers rather than using longer timeouts or weaker selectors.
5. The CSP used the full pathful `NEXT_PUBLIC_API_URL` as a `connect-src` source. Browser traces proved that `/auth/me` was blocked by CSP before reaching the mock/API. CSP generation now reduces the configured API URL to `URL.origin`, with a Vitest regression contract for pathful, portful, missing, and malformed inputs.

## Required evidence

The exact final PR head must pass fresh repository CI and the Media Library branch qualification workflow. At minimum this includes frozen dependency installation, Prisma generation/migrations, static formatting/lint/type gates, ML policy contracts, Media Library/API unit tests, API e2e tests, API and web production builds, mobile type-check, and Chromium accessibility/visual regression contracts.

The final browser smoke must also demonstrate that the emitted CSP permits the configured API origin and that authenticated `/auth/me` traffic is not blocked by Content Security Policy.

Do not mark the parent integration PR ready based on historical child-branch runs. Record the exact final SHA and successful workflow runs in PR #7 before changing draft status.

After this parent lands on `main`, Adventure PR #6 must be restacked on that exact `main` and its complete Adventure qualification pipeline rerun before it can be marked ready.
