import type { BrowserContext, Page, Route } from '@playwright/test';
import {
  AUTH_STORAGE_KEY,
  LEGACY_SESSION_STORAGE_KEY,
  serializePersistedAuthSession,
} from '../../src/lib/stores/auth-persist';
import type { AuthUser } from '../../src/lib/stores/auth-store';

export type { AuthUser };

export interface SeedAuthenticatedSessionOptions {
  user: AuthUser;
  token: string;
  routeAuthMe?: boolean;
}

function corsHeaders(route: Route) {
  const requestHeaders = route.request().headers();
  return {
    'access-control-allow-origin': requestHeaders.origin ?? 'http://localhost:3000',
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
 * same-origin page. No inline init script is used, so CSP behavior remains real.
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
    ([storageKey, legacyKey, authToken, persistedState]) => {
      window.localStorage.setItem('authToken', authToken);
      window.localStorage.setItem(storageKey, persistedState);
      window.localStorage.removeItem(legacyKey);
    },
    [AUTH_STORAGE_KEY, LEGACY_SESSION_STORAGE_KEY, token, persisted] as const
  );
}

export async function clearAuthenticatedSession(page: Page): Promise<void> {
  await page.goto('/login');
  await page.evaluate(
    ([storageKey, legacyKey]) => {
      window.localStorage.removeItem('authToken');
      window.localStorage.removeItem(storageKey);
      window.localStorage.removeItem(legacyKey);
    },
    [AUTH_STORAGE_KEY, LEGACY_SESSION_STORAGE_KEY] as const
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
  {
    latitude,
    longitude,
    accuracy = 100,
    origin = 'http://localhost:3000',
  }: RoughLocation
): Promise<void> {
  await context.grantPermissions(['geolocation'], { origin });
  await context.setGeolocation({ latitude, longitude, accuracy });
}
