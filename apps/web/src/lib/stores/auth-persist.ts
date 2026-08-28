import type { AuthUser } from './auth-store';

export const AUTH_STORAGE_KEY = 'woof-auth-storage';
export const LEGACY_SESSION_STORAGE_KEY = 'woof-session-storage';

export type PersistedAuthSnapshot = {
  state: {
    user: AuthUser | null;
    token: string | null;
    isAuthenticated: boolean;
  };
  version: 0;
};

/** Serializes the Zustand persist payload written by production auth storage. */
export function serializePersistedAuthSession(user: AuthUser, token: string): string {
  const payload: PersistedAuthSnapshot = {
    state: {
      user,
      token,
      isAuthenticated: true,
    },
    version: 0,
  };
  return JSON.stringify(payload);
}
