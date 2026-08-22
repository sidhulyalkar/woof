# Pet Media Library release qualification

This document is the merge checkpoint for the integrated pet-intelligence / Health Lens / Behavior Vision / private Media Library branch used as the parent of the Adventure System.

## Release invariants

- Private pet media remains private by default and signed upload/download URLs are never persisted as durable asset data.
- Browser authentication and the Media Library use the same canonical `useAuthStore`; an authenticated owner's pet list must enable the library query without a second session store.
- Media upload tests inspect Prisma arguments structurally so legitimate `bigint` byte counts remain supported while signed URLs and public visibility remain absent.
- Behavior Vision fails closed and its personal-profile tests must parse and execute before release.
- Health Lens emergency behavior remains outside game/reward mechanics.
- ML promoted-mode behavior retains artifact attestation and fallback guarantees.
- One-run branch-construction payloads, publishing workflows, and finalize scripts are not part of the release candidate.

## Required evidence

The exact final PR head must pass fresh repository CI and the Media Library branch qualification workflow. At minimum this includes frozen dependency installation, Prisma generation/migrations, static formatting/type gates, Media Library/API unit tests, API and web production builds, mobile type-check, and Chromium accessibility/visual regression contracts.

Do not mark the parent integration PR ready based on historical child-branch runs. Record the exact final SHA and successful workflow runs in PR #7 before changing draft status.

After this parent lands on `main`, Adventure PR #6 must be restacked on that exact `main` and its complete Adventure qualification pipeline rerun before it can be marked ready.
