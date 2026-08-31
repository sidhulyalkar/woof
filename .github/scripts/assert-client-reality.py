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
middleware = WEB / "src" / "middleware.ts"
transport = WEB / "src" / "lib" / "security" / "transport.ts"
transport_test = WEB / "src" / "lib" / "security" / "transport.test.ts"
persist = WEB / "src" / "lib" / "stores" / "auth-persist.ts"
auth_store = WEB / "src" / "lib" / "stores" / "auth-store.ts"
auth_store_test = WEB / "src" / "lib" / "stores" / "auth-store.test.ts"
legacy_projection = WEB / "src" / "store" / "session.ts"
api_client = WEB / "src" / "lib" / "api" / "client.ts"
api_client_test = WEB / "src" / "lib" / "api" / "client.test.ts"
api_hooks = WEB / "src" / "lib" / "api" / "hooks.ts"
auth_guard = WEB / "src" / "components" / "auth-guard.tsx"
auth_guard_test = WEB / "src" / "components" / "auth-guard.test.tsx"
helper = WEB / "e2e" / "support" / "session.ts"
auth_spec = WEB / "e2e" / "auth.spec.ts"
workflow = ROOT / ".github" / "workflows" / "client-reality-ci.yml"
release_polish_workflow = ROOT / ".github" / "workflows" / "dogos-release-polish-ci.yml"
production_lifecycle = ROOT / ".github" / "scripts" / "playwright-production-web.sh"

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
    middleware,
    transport,
    transport_test,
    persist,
    auth_store,
    auth_store_test,
    api_client,
    api_client_test,
    api_hooks,
    auth_guard,
    auth_guard_test,
    helper,
    auth_spec,
    workflow,
    release_polish_workflow,
    production_lifecycle,
    *matrix_suites,
    *shared_fixture_suites,
]:
    if not path.is_file():
        raise SystemExit(f"required client-reality source missing: {path.relative_to(ROOT)}")

if legacy_projection.exists():
    raise SystemExit("apps/web/src/store/session.ts: retired client session adapter must not exist")

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
    "onRehydrateStorage",
]:
    require(auth_store, marker)
for marker in ["localStorage.setItem(", "hasHydrated:", "setHasHydrated"]:
    reject(auth_store, marker)

for marker in [
    "useAuthStore",
    "getCanonicalAccessToken",
    "return useAuthStore.getState().token",
    "clearStaleSessionAfterUnauthorized",
    "auth.logout()",
]:
    require(api_client, marker)
for marker in ["localStorage.getItem(", "localStorage.setItem(", "localStorage.removeItem("]:
    reject(api_client, marker)

require(api_hooks, "useAuthStore")
require(api_hooks, "auth.updateUser({ pets: [...currentPets, pet] })")
reject(api_hooks, "useSessionStore")
reject(api_hooks, "refreshSession")

for marker in [
    "const token = useAuthStore((state) => state.token)",
    "useAuthStore.persist.onHydrate(",
    "useAuthStore.persist.onFinishHydration(",
    "useAuthStore.persist.hasHydrated()",
    "useAuthStore.persist.rehydrate()",
    "const verifiedToken = useRef<string | null>(null)",
    "const candidateToken = token",
    "authApi",
    ".me()",
    "useAuthStore.getState().token !== candidateToken",
    "verifiedToken.current = candidateToken",
    "verifiedToken.current !== token",
    "setAuth(user, candidateToken)",
    "router.replace('/login')",
]:
    require(auth_guard, marker)
for marker in [
    "state.hasHydrated",
    "setHasHydrated",
    "localStorage.getItem(",
    "localStorage.setItem(",
    "localStorage.removeItem(",
    "if (isAuthenticated)",
]:
    reject(auth_guard, marker)

for marker in [
    "uses the real persistence lifecycle before deciding an unauthenticated protected route",
    "keeps protected content closed while the canonical persisted token is verified",
    "refreshes canonical user state only after the server accepts the persisted token",
    "fails closed and retires canonical authority when server verification rejects the token",
    "useAuthStore.persist.rehydrate()",
]:
    require(auth_guard_test, marker)
for marker in ["localStorage.setItem('authToken'", 'localStorage.setItem("authToken"']:
    reject(auth_guard_test, marker)

for path in [auth_store_test, api_client_test]:
    require(path, "LEGACY_SESSION_STORAGE_KEY")
    require(path, "LEGACY_RAW_AUTH_TOKEN_KEY")
for marker in [
    "reports the real Zustand hydration lifecycle instead of storing a shadow hydration flag",
    "useAuthStore.persist.onHydrate(",
    "useAuthStore.persist.onFinishHydration(",
    "useAuthStore.persist.hasHydrated()",
    "useAuthStore.persist.rehydrate()",
]:
    require(auth_store_test, marker)
require(api_client_test, "getCanonicalAccessToken")
require(api_client_test, "clearStaleSessionAfterUnauthorized")

for marker in [
    "normalized === 'localhost'",
    "normalized === '127.0.0.1'",
    "normalized === '::1'",
    "nodeEnv === 'production'",
    "forwardedProto !== 'https'",
    "!isLoopbackHostname(hostname)",
]:
    require(transport, marker)
for marker in [
    "requires HTTPS for a public production host",
    "accepts a public production request already proven HTTPS by the proxy",
    "allows HTTP only for local production qualification",
]:
    require(transport_test, marker)
for marker in [
    "shouldRedirectToHttps",
    "hostname: request.nextUrl.hostname",
    "const redirectUrl = request.nextUrl.clone()",
    "redirectUrl.protocol = 'https:'",
]:
    require(middleware, marker)
reject(middleware, "`https://${request.headers.get('host')}${request.nextUrl.pathname}`")

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
    "await page.goto('/login', { waitUntil: 'networkidle' })",
    "window.localStorage.setItem(storageKey, persistedState)",
    "window.localStorage.removeItem(legacySessionKey)",
    "window.localStorage.removeItem(legacyRawTokenKey)",
    "window.localStorage.getItem(storageKey)",
    "Canonical authenticated session was not retained",
    "Every caller",
    "authenticated cold start",
]:
    require(helper, marker)
for marker in [
    "addInitScript",
    "page.reload(",
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
release_polish_spec = WEB / "e2e" / "release-polish.spec.ts"
require(release_polish_spec, "E2E_ORIGIN")
reject(release_polish_spec, "http://localhost:3000")

for marker in [
    "name: 'chromium'",
    "name: 'Mobile Chrome'",
    "name: 'firefox'",
    "name: 'webkit'",
    "['line']",
    "['html', { open: 'never' }]",
    "const loopbackBaseUrl = 'http://127.0.0.1:3000'",
    "baseURL: loopbackBaseUrl",
    "PLAYWRIGHT_EXTERNAL_SERVER",
    "webServer: externalServer",
    "'NODE_ENV=production pnpm --filter @woof/web start'",
    "'pnpm --filter @woof/web dev'",
    "url: loopbackBaseUrl",
]:
    require(config, marker)
reject(config, "http://localhost:3000")

for marker in [
    'BASE_URL="http://127.0.0.1:3000"',
    "nohup env NODE_ENV=production pnpm --filter @woof/web start",
    "--write-out '%{http_code}'",
    'if [[ "$status" == "200" ]]',
    "Production Web server exited before readiness.",
    "Production Web server did not return HTTP 200 within 45 seconds.",
    'kill -0 "$pid"',
    "start)",
    "stop)",
]:
    require(production_lifecycle, marker)
reject(production_lifecycle, "curl --fail")

# Root E2E intentionally keeps one full-stack login case out of the mocked suite.
# The reason must remain explicit so a reported skip is attributable rather than mysterious.
require(auth_spec, "test.skip('logs in with seeded credentials against the full local stack'")
require(auth_spec, "Integration-only test: intentionally skipped unless the seeded API is running.")

for marker in [
    "apps/web/src/middleware.ts",
    "apps/web/src/lib/security/transport.ts",
    "apps/web/src/lib/security/transport.test.ts",
    "src/lib/security/transport.test.ts",
    "src/components/auth-guard.tsx",
    "src/components/auth-guard.test.tsx",
    "PLAYWRIGHT_EXTERNAL_SERVER: '1'",
    "Build Web client for production-server qualification",
    "Start and prove production Web server",
    "bash .github/scripts/playwright-production-web.sh start",
    "bash .github/scripts/playwright-production-web.sh stop",
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
    "src/components/auth-guard.test.tsx",
]:
    require(workflow, marker)

# Release-browser evidence must compile the same public API origin it exercises at
# runtime. NEXT_PUBLIC_* values are baked by Next at build time, so a per-build
# override would qualify a different client than the browser fixture.
for marker in [
    "apps/web/src/lib/stores/auth-store.ts",
    "apps/web/src/lib/stores/auth-store.test.ts",
    "NEXT_PUBLIC_API_URL: http://127.0.0.1:59999/api/v1",
    "Build Web app for production-server browser qualification",
    "run: pnpm --filter @woof/web build",
    "PLAYWRIGHT_EXTERNAL_SERVER: '1'",
]:
    require(release_polish_workflow, marker)
reject(release_polish_workflow, "https://api.example.com/api/v1")

print("Client reality contract preserves server-verified auth, real persistence hydration, settled side-effect-free fixture seeding, HTTPS policy, and build-identical browser evidence.")
