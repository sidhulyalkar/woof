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

shared_fixture_suites = [
    WEB / "e2e" / "trust-discovery.spec.ts",
    WEB / "e2e" / "release-polish.spec.ts",
    WEB / "e2e" / "behavior-moments-shadow.spec.ts",
    WEB / "e2e" / "library-regression.spec.ts",
]

for path in [
    config,
    persist,
    auth_store,
    auth_store_test,
    api_client,
    api_client_test,
    api_hooks,
    helper,
    auth_spec,
    workflow,
    *matrix_suites,
    *shared_fixture_suites,
]:
    if not path.is_file():
        raise SystemExit(f"required client-reality source missing: {path.relative_to(ROOT)}")

if legacy_projection.exists():
    raise SystemExit(
        "apps/web/src/store/session.ts: retired client session adapter must not exist"
    )

for marker in [
    "AUTH_STORAGE_KEY = 'woof-auth-storage'",
    "LEGACY_SESSION_STORAGE_KEY = 'woof-session-storage'",
    "LEGACY_RAW_AUTH_TOKEN_KEY = 'authToken'",
    "AUTH_PERSIST_VERSION = 0",
    "serializePersistedAuthSession",
]:
    require(persist, marker)

for marker in [
    "AUTH_PERSIST_VERSION",
    "AUTH_STORAGE_KEY",
    "LEGACY_RAW_AUTH_TOKEN_KEY",
    "LEGACY_SESSION_STORAGE_KEY",
    "retireLegacyBrowserAuth",
    "localStorage.removeItem(LEGACY_SESSION_STORAGE_KEY)",
    "localStorage.removeItem(LEGACY_RAW_AUTH_TOKEN_KEY)",
    "name: AUTH_STORAGE_KEY",
    "version: AUTH_PERSIST_VERSION",
]:
    require(auth_store, marker)
reject(auth_store, "localStorage.setItem(")

for marker in [
    "useAuthStore",
    "getCanonicalAccessToken",
    "return useAuthStore.getState().token",
    "clearStaleSessionAfterUnauthorized",
    "auth.logout()",
]:
    require(api_client, marker)
for marker in [
    "localStorage.getItem(",
    "localStorage.setItem(",
    "localStorage.removeItem(",
]:
    reject(api_client, marker)

require(api_hooks, "useAuthStore")
require(api_hooks, "auth.updateUser({ pets: [...currentPets, pet] })")
reject(api_hooks, "useSessionStore")
reject(api_hooks, "refreshSession")

for path in [auth_store_test, api_client_test]:
    require(path, "LEGACY_SESSION_STORAGE_KEY")
    require(path, "LEGACY_RAW_AUTH_TOKEN_KEY")
require(api_client_test, "getCanonicalAccessToken")
require(api_client_test, "clearStaleSessionAfterUnauthorized")

# No production source may retain the retired adapter or either historical raw
# storage literal. auth-persist.ts is the one constant owner so cleanup remains
# explicit without letting stale state regain authority.
for source in (WEB / "src").rglob("*"):
    if source.suffix not in {".ts", ".tsx"} or source == persist:
        continue
    if ".test." in source.name or ".spec." in source.name:
        continue
    for marker in [
        "@/store/session",
        "useSessionStore",
        "woof-session-storage",
        "'authToken'",
        '"authToken"',
    ]:
        reject(source, marker)

for marker in [
    "serializePersistedAuthSession",
    "LEGACY_RAW_AUTH_TOKEN_KEY",
    "await page.goto('/login')",
    "window.localStorage.setItem(storageKey, persistedState)",
    "window.localStorage.removeItem(legacySessionKey)",
    "window.localStorage.removeItem(legacyRawTokenKey)",
    "await page.reload({ waitUntil: 'domcontentloaded' })",
]:
    require(helper, marker)
for marker in [
    "addInitScript",
    "localStorage.setItem('authToken'",
    'localStorage.setItem("authToken"',
]:
    reject(helper, marker)

for suite in [*matrix_suites, *shared_fixture_suites]:
    require(suite, "seedAuthenticatedSession")
    for marker in [
        "addInitScript",
        "woof-auth-storage",
        "woof-session-storage",
        "localStorage.setItem('authToken'",
        'localStorage.setItem("authToken"',
    ]:
        reject(suite, marker)

require(WEB / "e2e" / "trust-discovery.spec.ts", "grantRoughLocation")

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
    "e2e/trust-discovery.spec.ts",
    "e2e/release-polish.spec.ts",
    "e2e/behavior-moments-shadow.spec.ts",
    "e2e/library-regression.spec.ts",
    "playwright install --with-deps ${{ matrix.browser }}",
    "src/lib/stores/auth-persist.test.ts",
    "src/lib/stores/auth-store.test.ts",
    "src/lib/api/client.test.ts",
]:
    require(workflow, marker)

print("Client reality contract preserves named browser evidence and one session authority.")
