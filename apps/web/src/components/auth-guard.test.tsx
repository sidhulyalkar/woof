import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useAuthStore } from '@/lib/stores/auth-store';
import { AuthGuard } from './auth-guard';

const { mockMe, mockReplace } = vi.hoisted(() => ({
  mockMe: vi.fn(),
  mockReplace: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: mockReplace }),
  usePathname: () => '/test',
}));

vi.mock('@/lib/api', () => ({
  authApi: {
    me: mockMe,
  },
}));

const persistedUser = {
  id: '123',
  handle: 'persisted-user',
  email: 'test@example.com',
};

function seedCanonicalCandidate() {
  useAuthStore.setState({
    user: persistedUser,
    token: 'persisted-token',
    isAuthenticated: true,
    isLoading: false,
  });
}

describe('AuthGuard', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    localStorage.clear();
    mockMe.mockResolvedValue({
      id: '123',
      handle: 'verified-user',
      email: 'test@example.com',
    });
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
    await useAuthStore.persist.rehydrate();
  });

  it('uses the real persistence lifecycle before deciding an unauthenticated protected route', async () => {
    expect(useAuthStore.persist.hasHydrated()).toBe(true);

    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith('/login');
    });
    expect(mockMe).not.toHaveBeenCalled();
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
  });

  it('keeps protected content closed while the canonical persisted token is verified', () => {
    seedCanonicalCandidate();
    mockMe.mockImplementation(() => new Promise(() => undefined));

    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>
    );

    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
    expect(mockMe).toHaveBeenCalledTimes(1);
  });

  it('redirects a hydrated protected route with no canonical token to login', async () => {
    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith('/login');
    });
    expect(mockMe).not.toHaveBeenCalled();
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
  });

  it('refreshes canonical user state only after the server accepts the persisted token', async () => {
    seedCanonicalCandidate();

    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>
    );

    await waitFor(() => {
      expect(screen.getByText('Protected Content')).toBeInTheDocument();
    });
    expect(mockMe).toHaveBeenCalledTimes(1);
    expect(useAuthStore.getState()).toMatchObject({
      user: {
        id: '123',
        handle: 'verified-user',
        email: 'test@example.com',
      },
      token: 'persisted-token',
      isAuthenticated: true,
    });
    expect(useAuthStore.persist.hasHydrated()).toBe(true);
  });

  it('fails closed and retires canonical authority when server verification rejects the token', async () => {
    seedCanonicalCandidate();
    mockMe.mockRejectedValue(new Error('revoked session'));

    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith('/login');
    });
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
    expect(useAuthStore.getState()).toMatchObject({
      user: null,
      token: null,
      isAuthenticated: false,
    });
  });
});
