import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { LEGACY_RAW_AUTH_TOKEN_KEY, LEGACY_SESSION_STORAGE_KEY } from './auth-persist';
import { useAuthStore } from './auth-store';

describe('Auth Store', () => {
  beforeEach(() => {
    localStorage.clear();
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      hasHydrated: true,
    });
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('should initialize with unauthenticated state once hydration is complete', () => {
    const { user, token, isAuthenticated, hasHydrated } = useAuthStore.getState();
    expect(user).toBeNull();
    expect(token).toBeNull();
    expect(isAuthenticated).toBe(false);
    expect(hasHydrated).toBe(true);
  });

  it('tracks persistence hydration separately from authentication authority', () => {
    useAuthStore.getState().setHasHydrated(false);

    expect(useAuthStore.getState()).toMatchObject({
      token: null,
      isAuthenticated: false,
      hasHydrated: false,
    });

    useAuthStore.getState().setHasHydrated(true);
    expect(useAuthStore.getState().hasHydrated).toBe(true);
  });

  it('should set auth correctly and retire historical browser auth mirrors', () => {
    const mockUser = { id: '123', handle: 'testuser', email: 'test@example.com' };
    const mockToken = 'mock-jwt-token';
    localStorage.setItem(LEGACY_SESSION_STORAGE_KEY, '{"stale":true}');
    localStorage.setItem(LEGACY_RAW_AUTH_TOKEN_KEY, 'stale-raw-token');

    useAuthStore.getState().setAuth(mockUser, mockToken);

    const { user, token, isAuthenticated, hasHydrated } = useAuthStore.getState();
    expect(user).toEqual(mockUser);
    expect(token).toBe(mockToken);
    expect(isAuthenticated).toBe(true);
    expect(hasHydrated).toBe(true);
    expect(localStorage.getItem(LEGACY_RAW_AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(LEGACY_SESSION_STORAGE_KEY)).toBeNull();
  });

  it('should logout correctly and fail closed across historical browser auth', () => {
    const mockUser = { id: '123', handle: 'testuser', email: 'test@example.com' };
    const mockToken = 'mock-jwt-token';
    useAuthStore.getState().setAuth(mockUser, mockToken);
    localStorage.setItem(LEGACY_SESSION_STORAGE_KEY, '{"stale":true}');
    localStorage.setItem(LEGACY_RAW_AUTH_TOKEN_KEY, 'stale-raw-token');

    useAuthStore.getState().logout();

    const { user, token, isAuthenticated, hasHydrated } = useAuthStore.getState();
    expect(user).toBeNull();
    expect(token).toBeNull();
    expect(isAuthenticated).toBe(false);
    expect(hasHydrated).toBe(true);
    expect(localStorage.getItem(LEGACY_RAW_AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(LEGACY_SESSION_STORAGE_KEY)).toBeNull();
  });

  it('should update user correctly', () => {
    const mockUser = { id: '123', handle: 'testuser', email: 'test@example.com' };
    const mockToken = 'mock-jwt-token';
    useAuthStore.getState().setAuth(mockUser, mockToken);
    useAuthStore.getState().updateUser({ handle: 'newhandle' });
    const { user } = useAuthStore.getState();
    expect(user?.handle).toBe('newhandle');
    expect(user?.email).toBe('test@example.com');
  });
});
