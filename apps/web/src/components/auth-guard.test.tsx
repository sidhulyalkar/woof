import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useAuthStore } from '@/lib/stores/auth-store';
import { AuthGuard } from './auth-guard';

const mockReplace = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: mockReplace }),
  usePathname: () => '/test',
}));

vi.mock('@/lib/api', () => ({
  authApi: {
    me: vi.fn(() =>
      Promise.resolve({
        id: '123',
        handle: 'testuser',
        email: 'test@example.com',
      }),
    ),
  },
}));

describe('AuthGuard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  });

  it('shows a loading state while session hydration runs', () => {
    render(
      <AuthGuard>
        <div>Protected Content</div>
      </AuthGuard>,
    );

    expect(screen.getByRole('status')).toBeInTheDocument();
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
  });
});
