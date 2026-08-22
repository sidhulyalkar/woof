# Pet Media Library release qualification

This document is the merge checkpoint for the integrated pet-intelligence / Health Lens / Behavior Vision / private Media Library branch used as the parent of the Adventure System.

## Release invariants

- Private pet media remains private by default and signed upload/download URLs are never persisted as durable asset data.
- Browser authentication and the Media Library use the same canonical `useAuthStore`; an authenticated owner's pet list must enable the library query without a second session store.
- Protected-browser tests seed the auth token before application code runs, so session bootstrap is deterministic rather than relying on a navigation race.
- Anonymous login rejection remains form-level feedback. The global HTTP 401 handler clears and redirects only when the browser actually has an authenticated token to invalidate.
- The web Content Security Policy derives `connect-src` from the configured API **origin**, not a pathful API base. A value such as `https://api.example.com/api/v1` must allow descendant endpoints such as `/api/v1/auth/me` without broadening the policy beyond `https://api.example.com`.
- Library loading, error, empty, uploading, albums, selection, and populated-grid states expose stable browser contracts and an API/storage failure must render an actionable retry state rather than masquerading as an empty library.
- Media upload tests inspect Prisma arguments structurally so legitimate `bigint` byte counts remain supported while signed URLs and public visibility remain absent.
- The mobile Media Library explicitly owns its Expo SDK 54 file-system/sharing dependencies, and native media file metadata uses the SDK-compatible legacy `getInfoAsync(uri)` contract.
- Mobile pet lists remain owner-scoped; proximity discovery fails closed until a dedicated privacy-preserving backend proximity contract exists.
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
6. The Library regression suite expected explicit UI-state contracts that the page did not expose, and the page had no dedicated query-error branch. The Library now distinguishes loading, error, empty, uploading, albums, selection, and populated-grid states, provides a bounded automatic retry plus a user-visible `Try again` action, and preserves per-asset selectors for deterministic interaction tests.
7. The browser auth helper previously wrote `localStorage` only after first visiting `/login`, creating a race where `AuthGuard` could remain in `Checking your Woof session`. Tests now install the auth token with `addInitScript` before protected navigation so the first application frame sees the intended authenticated state.
8. The global Axios response interceptor treated every 401, including an expected anonymous login rejection, as an expired authenticated session. It now redirects only when an auth token existed, allowing invalid-credential errors to remain visible on the login form.
9. Native qualification exposed missing Expo file-system/sharing dependencies plus stale mobile API envelope assumptions. The parent now declares the SDK-compatible dependencies, uses the current file metadata signature, types goal responses at the unwrapped API boundary, consumes feed/pet envelopes correctly, and keeps nearby-pet discovery fail-closed rather than inferring proximity from broad pet data.

## Required evidence

The exact final PR head must pass fresh repository CI and the Media Library branch qualification workflow. At minimum this includes frozen dependency installation, Prisma generation/migrations, static formatting/lint/type gates, ML policy contracts, Media Library/API unit tests, API e2e tests, API and web production builds, mobile type-check, and Chromium accessibility/visual regression contracts.

The final browser smoke must also demonstrate that the emitted CSP permits the configured API origin, that authenticated `/auth/me` traffic is not blocked by Content Security Policy, and that an anonymous `/auth/login` 401 remains visible as form feedback rather than triggering a global session redirect.

Do not mark the parent integration PR ready based on historical child-branch runs. Record the exact final SHA and successful workflow runs in PR #7 before changing draft status.

After this parent lands on `main`, Adventure PR #6 must be restacked on that exact `main` and its complete Adventure qualification pipeline rerun before it can be marked ready.
