import type { BrowserContext, Page, Route } from '@playwright/test';
import {
  AUTH_STORAGE_KEY,
  LEGACY_RAW_AUTH_TOKEN_KEY,
  LEGACY_SESSION_STORAGE_KEY,
  serializePersistedAuthSession,
} from '../../src/lib/stores/auth-persist';
import type { AuthUser } from '../../src/lib/stores/auth-store';

export type { AuthUser };

export const E2E_ORIGIN = 'http://127.0.0.1:3000';

export interface SeedAuthenticatedSessionOptions {
  user: AuthUser;
  token: string;
  routeAuthMe?: boolean;
}

function corsHeaders(route: Route) {
  const requestHeaders = route.request().headers();
  return {
    'access-control-allow-origin': requestHeaders.origin ?? E2E_ORIGIN,
    'access-control-allow-methods': 'GET,POST,PUT,PATCH,DELETE,OPTIONS',
    'access-control-allow-headers':
      requestHeaders['access-control-request-headers'] ?? 'authorization,content-type',
    vary: 'Origin',
  };
}

async function fulfillAuthMe(route: Route, user: AuthUser) {
  if (route.request().method() === 'OPTIONS') {
    await route.fulfill({ status: 204, headers: corsHeaders(route), body: '' });
    return;
  }
  await route.fulfill({
    status: 200,
    headers: { ...corsHeaders(route), 'content-type': 'application/json' },
    body: JSON.stringify(user),
  });
}

/**
 * Seeds exactly the persisted client session production owns from a real
 * same-origin page. No inline init script or duplicate raw-token mirror is used,
 * so CSP behavior and token authority remain representative of production.
 * Reload after the write so every engine boots the application from the same
 * persisted-state lifecycle instead of relying on an already-mounted store to
 * notice a same-document localStorage mutation.
 */
export async function seedAuthenticatedSession(
  page: Page,
  { user, token, routeAuthMe = true }: SeedAuthenticatedSessionOptions
): Promise<void> {
  if (routeAuthMe) {
    await page.route('**/auth/me', (route) => fulfillAuthMe(route, user));
  }

  const persisted = serializePersistedAuthSession(user, token);
  await page.goto('/login');
  await page.evaluate(
    ([storageKey, legacySessionKey, legacyRawTokenKey, persistedState]) => {
      window.localStorage.setItem(storageKey, persistedState);
      window.localStorage.removeItem(legacySessionKey);
      window.localStorage.removeItem(legacyRawTokenKey);
    },
    [AUTH_STORAGE_KEY, LEGACY_SESSION_STORAGE_KEY, LEGACY_RAW_AUTH_TOKEN_KEY, persisted] as const
  );
  await page.reload({ waitUntil: 'domcontentloaded' });
}

export async function clearAuthenticatedSession(page: Page): Promise<void> {
  await page.goto('/login');
  await page.evaluate(
    ([storageKey, legacySessionKey, legacyRawTokenKey]) => {
      window.localStorage.removeItem(storageKey);
      window.localStorage.removeItem(legacySessionKey);
      window.localStorage.removeItem(legacyRawTokenKey);
    },
    [AUTH_STORAGE_KEY, LEGACY_SESSION_STORAGE_KEY, LEGACY_RAW_AUTH_TOKEN_KEY] as const
  );
}

export interface RoughLocation {
  latitude: number;
  longitude: number;
  accuracy?: number;
  origin?: string;
}

/** Grants geolocation through Playwright's browser context, preserving CSP. */
export async function grantRoughLocation(
  context: BrowserContext,
  { latitude, longitude, accuracy = 100, origin = E2E_ORIGIN }: RoughLocation
): Promise<void> {
  await context.grantPermissions(['geolocation'], { origin });
  await context.setGeolocation({ latitude, longitude, accuracy });
}
