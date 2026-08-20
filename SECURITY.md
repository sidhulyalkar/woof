# Security Policy

Woof is a portfolio and research project that explores location-aware social coordination between pet owners. Because the product model includes accounts, precise location, routes, messaging, uploads and real-world meetups, security and privacy issues should be treated seriously even in demo environments.

## Supported code

Security work should target the canonical applications under `apps/`, `packages/`, `ml/`, `infra/` and `n8n/`. Material under `legacy/` and `docs/archive/` is preserved for provenance and is not an active deployment target.

## Reporting a vulnerability

Please do not open a public issue containing exploit details, credentials, private user information or a reproducible attack against a deployed environment.

Use GitHub's private vulnerability reporting / Security Advisory flow for this repository when available. Include:

- affected component and version or commit,
- impact,
- reproduction steps,
- relevant request/response details with secrets removed,
- a suggested mitigation if you have one.

## Security boundaries that matter most

The highest-risk areas in Woof are:

1. authorization across user-owned pets, conversations, activities and meetup data,
2. precise home, route and live location exposure,
3. account/session security,
4. file upload validation and object access,
5. realtime room authorization,
6. abuse, blocking and reporting for real-world meetup flows,
7. service-to-service trust around ML and automation integrations.

## Production disclaimer

The repository contains production-oriented controls such as JWT authentication, request validation, Helmet, origin configuration, rate limiting and monitoring hooks. These components do not by themselves make the current project suitable for unsupervised public deployment with sensitive real-world data.

Before a production launch, the project should complete threat modeling, authorization testing, secrets management review, dependency/container scanning, privacy and deletion workflows, backup/restore drills, abuse tooling and deployment-specific security-header verification.
