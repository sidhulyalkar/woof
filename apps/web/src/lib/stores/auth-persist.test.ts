import { describe, expect, it } from 'vitest';
import {
  AUTH_PERSIST_VERSION,
  AUTH_STORAGE_KEY,
  LEGACY_SESSION_STORAGE_KEY,
  serializePersistedAuthSession,
} from './auth-persist';

describe('persisted auth contract', () => {
  it('owns one canonical storage key and an explicit schema version', () => {
    expect(AUTH_STORAGE_KEY).toBe('woof-auth-storage');
    expect(LEGACY_SESSION_STORAGE_KEY).toBe('woof-session-storage');
    expect(AUTH_PERSIST_VERSION).toBe(0);
  });

  it('serializes the exact production Zustand payload browser tests seed', () => {
    const user = {
      id: 'user-1',
      handle: 'trailpaws',
      email: 'trailpaws@example.com',
      pets: [{ id: 'pet-1', name: 'Mochi', species: 'DOG' }],
    };
    const token = 'browser-test-token';

    expect(JSON.parse(serializePersistedAuthSession(user, token))).toEqual({
      state: {
        user,
        token,
        isAuthenticated: true,
      },
      version: 0,
    });
  });
});
