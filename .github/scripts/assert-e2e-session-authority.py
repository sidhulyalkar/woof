#!/usr/bin/env python3
"""Fail closed when browser auth fixtures drift from production persisted auth."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "apps" / "web"

TARGET_SUITES = [
    WEB / "e2e" / "release-polish.spec.ts",
    WEB / "e2e" / "trust-discovery.spec.ts",
    WEB / "e2e" / "library-regression.spec.ts",
    WEB / "e2e" / "behavior-moments-shadow.spec.ts",
]


def require(path: Path, marker: str) -> None:
    if marker not in path.read_text():
        raise SystemExit(f"{path.relative_to(ROOT)}: missing required marker {marker!r}")


def reject(path: Path, marker: str) -> None:
    if marker in path.read_text():
        raise SystemExit(f"{path.relative_to(ROOT)}: forbidden auth drift marker {marker!r}")


persist = WEB / "src" / "lib" / "stores" / "auth-persist.ts"
auth_store = WEB / "src" / "lib" / "stores" / "auth-store.ts"
helper = WEB / "e2e" / "support" / "session.ts"

for path in [persist, auth_store, helper, *TARGET_SUITES]:
    if not path.is_file():
        raise SystemExit(f"required session authority source missing: {path.relative_to(ROOT)}")

for marker in [
    "AUTH_STORAGE_KEY = 'woof-auth-storage'",
    "LEGACY_SESSION_STORAGE_KEY = 'woof-session-storage'",
    "AUTH_PERSIST_VERSION = 0",
    "serializePersistedAuthSession",
]:
    require(persist, marker)

for marker in [
    "AUTH_PERSIST_VERSION",
    "AUTH_STORAGE_KEY",
    "name: AUTH_STORAGE_KEY",
    "version: AUTH_PERSIST_VERSION",
]:
    require(auth_store, marker)

# The canonical store must not start deleting the legacy product store until its
# remaining production consumers have been migrated in a separate release.
reject(auth_store, "LEGACY_SESSION_STORAGE_KEY")
reject(auth_store, "woof-session-storage")

for marker in [
    "AUTH_STORAGE_KEY",
    "LEGACY_SESSION_STORAGE_KEY",
    "serializePersistedAuthSession",
    "await page.goto('/login')",
    "await page.evaluate",
    "window.localStorage.setItem(storageKey, persistedState)",
    "window.localStorage.removeItem(legacyKey)",
]:
    require(helper, marker)

reject(helper, "addInitScript")

for suite in TARGET_SUITES:
    require(suite, "seedAuthenticatedSession")
    for marker in [
        "addInitScript",
        "woof-auth-storage",
        "woof-session-storage",
        "localStorage.setItem('authToken'",
        'localStorage.setItem("authToken"',
    ]:
        reject(suite, marker)

trust = WEB / "e2e" / "trust-discovery.spec.ts"
require(trust, "grantRoughLocation")
for marker in ["navigator.geolocation", "getCurrentPosition =", "watchPosition ="]:
    reject(trust, marker)

print("E2E session authority contract is production-aligned and injection-free.")
