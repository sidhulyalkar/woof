import type { BrowserContext, Page } from '@playwright/test';
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
}

/**
 * Seeds the same persisted client session production writes, from a real
 * same-origin page (no addInitScript / CSP-inline bootstrap).
 */
export async function seedAuthenticatedSession(
  page: Page,
  { user, token }: SeedAuthenticatedSessionOptions,
): Promise<void> {
  const persisted = serializePersistedAuthSession(user, token);
  await page.goto('/login');
  await page.evaluate(
    ([storageKey, legacyKey, authToken, persistedState]) => {
      window.localStorage.setItem('authToken', authToken);
      window.localStorage.setItem(storageKey, persistedState);
      window.localStorage.removeItem(legacyKey);
    },
    [AUTH_STORAGE_KEY, LEGACY_SESSION_STORAGE_KEY, token, persisted] as const,
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
    [AUTH_STORAGE_KEY, LEGACY_SESSION_STORAGE_KEY] as const,
  );
}

export interface RoughLocation {
  latitude: number;
  longitude: number;
  accuracy?: number;
}

/** Grants geolocation through Playwright's browser context (CSP-safe). */
export async function grantRoughLocation(
  context: BrowserContext,
  { latitude, longitude, accuracy = 100 }: RoughLocation,
): Promise<void> {
  await context.grantPermissions(['geolocation']);
  await context.setGeolocation({ latitude, longitude, accuracy });
}
