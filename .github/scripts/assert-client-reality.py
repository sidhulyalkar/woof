#!/usr/bin/env python3
"""Fail closed when the qualified client matrix loses browser or session authority."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "apps" / "web"


def require(path: Path, marker: str) -> None:
    text = path.read_text()
    if marker not in text:
        raise SystemExit(f"{path.relative_to(ROOT)}: missing required marker {marker!r}")


def reject(path: Path, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        raise SystemExit(f"{path.relative_to(ROOT)}: forbidden client-reality marker {marker!r}")


config = WEB / "playwright.config.ts"
persist = WEB / "src" / "lib" / "stores" / "auth-persist.ts"
auth_store = WEB / "src" / "lib" / "stores" / "auth-store.ts"
auth_store_test = WEB / "src" / "lib" / "stores" / "auth-store.test.ts"
legacy_projection = WEB / "src" / "store" / "session.ts"
api_client = WEB / "src" / "lib" / "api" / "client.ts"
api_client_test = WEB / "src" / "lib" / "api" / "client.test.ts"
api_hooks = WEB / "src" / "lib" / "api" / "hooks.ts"
helper = WEB / "e2e" / "support" / "session.ts"
auth_spec = WEB / "e2e" / "auth.spec.ts"
workflow = ROOT / ".github" / "workflows" / "client-reality-ci.yml"

matrix_suites = [
    WEB / "e2e" / "companion-onramp.spec.ts",
    WEB / "e2e" / "navigation-spine.spec.ts",
    WEB / "e2e" / "caregiver-authority.spec.ts",
]

for path in [
    config,
    persist,
    auth_store,
    auth_store_test,
    legacy_projection,
    api_client,
    api_client_test,
    api_hooks,
    helper,
    auth_spec,
    workflow,
    *matrix_suites,
]:
    if not path.is_file():
        raise SystemExit(f"required client-reality source missing: {path.relative_to(ROOT)}")

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
    "LEGACY_SESSION_STORAGE_KEY",
    "retireLegacySessionStorage",
    "localStorage.removeItem(LEGACY_SESSION_STORAGE_KEY)",
    "name: AUTH_STORAGE_KEY",
    "version: AUTH_PERSIST_VERSION",
]:
    require(auth_store, marker)

# The historical module may remain temporarily as a UI alias projection, but it
# must never regain authentication or persistence authority.
for marker in [
    "useAuthStore",
    "Compatibility-only presentation projection",
    "projectUser",
    "projectPet",
]:
    require(legacy_projection, marker)
for marker in [
    "persist(",
    "refreshToken",
    "authToken",
    "/auth/me",
    "isAuthenticated",
    "refreshSession",
    "setSession",
    "clearSession",
    "localStorage",
]:
    reject(legacy_projection, marker)

for marker in [
    "useAuthStore",
    "clearStaleSessionAfterUnauthorized",
    "auth.logout()",
]:
    require(api_client, marker)
reject(api_client, "localStorage.removeItem('authToken')")

require(api_hooks, "useAuthStore")
require(api_hooks, "auth.updateUser({ pets: [...currentPets, pet] })")
reject(api_hooks, "useSessionStore")
reject(api_hooks, "refreshSession")

for path in [auth_store_test, api_client_test]:
    require(path, "LEGACY_SESSION_STORAGE_KEY")
require(api_client_test, "clearStaleSessionAfterUnauthorized")

# No production source may spell the retired persistence key directly. The one
# constant owner remains auth-persist.ts so stale-state cleanup can be explicit.
for source in (WEB / "src").rglob("*"):
    if source.suffix not in {".ts", ".tsx"} or source == persist:
        continue
    reject(source, "woof-session-storage")

for marker in [
    "serializePersistedAuthSession",
    "await page.goto('/login')",
    "window.localStorage.setItem(storageKey, persistedState)",
    "window.localStorage.removeItem(legacyKey)",
    "await page.reload({ waitUntil: 'domcontentloaded' })",
]:
    require(helper, marker)
reject(helper, "addInitScript")

for suite in matrix_suites:
    require(suite, "seedAuthenticatedSession")
    for marker in [
        "addInitScript",
        "woof-auth-storage",
        "woof-session-storage",
        "localStorage.setItem('authToken'",
        'localStorage.setItem("authToken"',
    ]:
        reject(suite, marker)

for marker in [
    "name: 'chromium'",
    "name: 'Mobile Chrome'",
    "name: 'firefox'",
    "name: 'webkit'",
    "['line']",
    "['html', { open: 'never' }]",
]:
    require(config, marker)

# Root E2E intentionally keeps one full-stack login case out of the mocked suite.
# The reason must remain explicit so a reported skip is attributable rather than mysterious.
require(auth_spec, "test.skip('logs in with seeded credentials against the full local stack'")
require(auth_spec, "Integration-only test: intentionally skipped unless the seeded API is running.")

for marker in [
    "project: chromium",
    "project: 'Mobile Chrome'",
    "project: firefox",
    "project: webkit",
    "browser: chromium",
    "browser: firefox",
    "browser: webkit",
    "e2e/auth.spec.ts",
    "e2e/companion-onramp.spec.ts",
    "e2e/navigation-spine.spec.ts",
    "e2e/caregiver-authority.spec.ts",
    "playwright install --with-deps ${{ matrix.browser }}",
    "src/lib/stores/auth-store.test.ts",
    "src/lib/api/client.test.ts",
]:
    require(workflow, marker)

print("Client reality contract preserves named browser evidence and one session authority.")
