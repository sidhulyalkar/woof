import { describe, expect, it } from 'vitest';
import {
  AUTH_STORAGE_KEY,
  LEGACY_SESSION_STORAGE_KEY,
  serializePersistedAuthSession,
} from './auth-persist';

describe('auth persist contract', () => {
  it('uses the production storage key and schema', () => {
    expect(AUTH_STORAGE_KEY).toBe('woof-auth-storage');
    expect(LEGACY_SESSION_STORAGE_KEY).toBe('woof-session-storage');

    const user = {
      id: 'user-1',
      handle: 'trailpaws',
      email: 'trailpaws@example.com',
      pets: [{ id: 'pet-1', name: 'Mochi', species: 'DOG' }],
    };
    const token = 'browser-test-token';

    const parsed = JSON.parse(serializePersistedAuthSession(user, token));
    expect(parsed).toEqual({
      state: {
        user,
        token,
        isAuthenticated: true,
      },
      version: 0,
    });
  });
});
