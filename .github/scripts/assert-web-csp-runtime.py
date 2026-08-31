#!/usr/bin/env python3
"""Fail closed if production nonce CSP stops authorizing the rendered Next runtime."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "apps" / "web"


def require(path: Path, marker: str) -> None:
    text = path.read_text()
    if marker not in text:
        raise SystemExit(f"{path.relative_to(ROOT)}: missing CSP runtime marker {marker!r}")


def reject(path: Path, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        raise SystemExit(f"{path.relative_to(ROOT)}: forbidden CSP runtime marker {marker!r}")


middleware = WEB / "src" / "middleware.ts"
layout = WEB / "src" / "app" / "layout.tsx"
auth_spec = WEB / "e2e" / "auth.spec.ts"

for path in [middleware, layout, auth_spec]:
    if not path.is_file():
        raise SystemExit(f"required CSP runtime source missing: {path.relative_to(ROOT)}")

for marker in [
    "const nonce = Buffer.from(crypto.randomUUID()).toString('base64')",
    "'nonce-${nonce}' 'strict-dynamic'",
    "const requestHeaders = new Headers(request.headers)",
    "requestHeaders.set('x-nonce', nonce)",
    "requestHeaders.set('Content-Security-Policy', cspHeader)",
    "NextResponse.next({",
    "headers: requestHeaders",
    "response.headers.set('Content-Security-Policy', cspHeader)",
]:
    require(middleware, marker)

# Response-only nonce CSP is not enough: Next must see the same CSP on the
# incoming render request or its framework scripts cannot inherit the nonce.
reject(middleware, "const response = NextResponse.next();")

for marker in [
    "import { headers } from 'next/headers'",
    "export default async function RootLayout",
    "await headers()",
    "nonce to framework, page, and inline runtime scripts during SSR",
]:
    require(layout, marker)

for marker in [
    "binds the response CSP nonce to the Next runtime script",
    "response.headers()['content-security-policy']",
    "csp.match(/'nonce-([^']+)'/)",
    "script[src^=\"/_next/static/\"]",
    "script.nonce",
    "expect(runtimeNonce).toBe(responseNonce)",
]:
    require(auth_spec, marker)

print(
    "Web CSP runtime contract preserves request-propagated nonces, dynamic App Router rendering, and browser attestation of the Next runtime script."
)
