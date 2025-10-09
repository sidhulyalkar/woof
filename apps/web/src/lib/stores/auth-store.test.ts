import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useAuthStore } from './auth-store';

describe('Auth Store', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();
    // Reset store state
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('should initialize with unauthenticated state', () => {
    const { user, token, isAuthenticated } = useAuthStore.getState();

    expect(user).toBeNull();
    expect(token).toBeNull();
    expect(isAuthenticated).toBe(false);
  });

  it('should set auth correctly', () => {
    const mockUser = {
      id: '123',
      handle: 'testuser',
      email: 'test@example.com',
    };
    const mockToken = 'mock-jwt-token';

    useAuthStore.getState().setAuth(mockUser, mockToken);

    const { user, token, isAuthenticated } = useAuthStore.getState();
    expect(user).toEqual(mockUser);
    expect(token).toBe(mockToken);
    expect(isAuthenticated).toBe(true);
    expect(localStorage.getItem('authToken')).toBe(mockToken);
  });

  it('should logout correctly', () => {
    const mockUser = {
      id: '123',
      handle: 'testuser',
      email: 'test@example.com',
    };
    const mockToken = 'mock-jwt-token';

    useAuthStore.getState().setAuth(mockUser, mockToken);
    useAuthStore.getState().logout();

    const { user, token, isAuthenticated } = useAuthStore.getState();
    expect(user).toBeNull();
    expect(token).toBeNull();
    expect(isAuthenticated).toBe(false);
    expect(localStorage.getItem('authToken')).toBeNull();
  });

  it('should update user correctly', () => {
    const mockUser = {
      id: '123',
      handle: 'testuser',
      email: 'test@example.com',
    };
    const mockToken = 'mock-jwt-token';

    useAuthStore.getState().setAuth(mockUser, mockToken);
    useAuthStore.getState().updateUser({ handle: 'newhandle' });

    const { user } = useAuthStore.getState();
    expect(user?.handle).toBe('newhandle');
    expect(user?.email).toBe('test@example.com');
  });
});
