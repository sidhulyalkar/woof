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

describe('AuthGuard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockMe.mockResolvedValue({
      id: '123',
      handle: 'testuser',
      email: 'test@example.com',
    });
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  });

  it('shows a loading state while a persisted session is being hydrated', () => {
    localStorage.setItem('authToken', 'persisted-token');
    mockMe.mockImplementation(() => new Promise(() => undefined));

    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>,
    );

    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
  });

  it('redirects unauthenticated visitors to login', async () => {
    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>,
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith('/login');
    });
    expect(mockMe).not.toHaveBeenCalled();
  });

  it('hydrates a valid persisted session before rendering protected content', async () => {
    localStorage.setItem('authToken', 'persisted-token');

    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>,
    );

    await waitFor(() => {
      expect(screen.getByText('Protected Content')).toBeInTheDocument();
    });
    expect(mockMe).toHaveBeenCalledTimes(1);
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
  });

  it('renders protected children for an authenticated session', async () => {
    useAuthStore.setState({
      user: { id: '123', handle: 'testuser', email: 'test@example.com' },
      token: 'mock-token',
      isAuthenticated: true,
      isLoading: false,
    });

    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>,
    );

    await waitFor(() => {
      expect(screen.getByText('Protected Content')).toBeInTheDocument();
    });
    expect(mockMe).not.toHaveBeenCalled();
  });
});
