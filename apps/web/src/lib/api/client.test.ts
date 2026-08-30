import { beforeEach, describe, expect, it } from 'vitest';
import { LEGACY_SESSION_STORAGE_KEY } from '@/lib/stores/auth-persist';
import { useAuthStore } from '@/lib/stores/auth-store';
import { clearStaleSessionAfterUnauthorized } from './client';

const user = { id: 'user-1', handle: 'trailpaws', email: 'trailpaws@example.com' };

describe('API unauthorized session boundary', () => {
  beforeEach(() => {
    localStorage.clear();
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  });

  it('clears canonical and historical browser auth state after an authenticated 401', () => {
    useAuthStore.getState().setAuth(user, 'access-token');
    localStorage.setItem(LEGACY_SESSION_STORAGE_KEY, '{"stale":true}');

    expect(clearStaleSessionAfterUnauthorized()).toBe(true);
    expect(localStorage.getItem('authToken')).toBeNull();
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
