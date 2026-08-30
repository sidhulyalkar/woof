import { beforeEach, describe, expect, it } from 'vitest';
import { LEGACY_RAW_AUTH_TOKEN_KEY, LEGACY_SESSION_STORAGE_KEY } from '@/lib/stores/auth-persist';
import { useAuthStore } from '@/lib/stores/auth-store';
import { clearStaleSessionAfterUnauthorized, getCanonicalAccessToken } from './client';

const user = { id: 'user-1', handle: 'trailpaws', email: 'trailpaws@example.com' };

describe('API canonical auth boundary', () => {
  beforeEach(() => {
    localStorage.clear();
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  });

  it('reads bearer authority from canonical state, never a conflicting historical raw key', () => {
    useAuthStore.setState({ token: 'canonical-token', isAuthenticated: true });
    localStorage.setItem(LEGACY_RAW_AUTH_TOKEN_KEY, 'stale-raw-token');

    expect(getCanonicalAccessToken()).toBe('canonical-token');
  });

  it('clears canonical and historical browser auth state after an authenticated 401', () => {
    useAuthStore.getState().setAuth(user, 'access-token');
    localStorage.setItem(LEGACY_SESSION_STORAGE_KEY, '{"stale":true}');
    localStorage.setItem(LEGACY_RAW_AUTH_TOKEN_KEY, 'stale-raw-token');

    expect(clearStaleSessionAfterUnauthorized()).toBe(true);
    expect(localStorage.getItem(LEGACY_RAW_AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(LEGACY_SESSION_STORAGE_KEY)).toBeNull();
    expect(useAuthStore.getState()).toMatchObject({
      user: null,
      token: null,
      isAuthenticated: false,
    });
  });

  it('does not manufacture a logout when no authenticated browser state exists', () => {
    expect(clearStaleSessionAfterUnauthorized()).toBe(false);
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });
});
