import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: string;
  handle: string;
  email: string;
  bio?: string;
  avatarUrl?: string;
  points?: number;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  setAuth: (user: User, token: string) => void;
  logout: () => void;
  updateUser: (user: Partial<User>) => void;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      setAuth: (user, token) => {
        // Store token in localStorage for API client interceptor
        if (typeof window !== 'undefined') {
          localStorage.setItem('authToken', token);
        }
        set({ user, token, isAuthenticated: true, isLoading: false });
      },

      logout: () => {
        // Remove token from localStorage
        if (typeof window !== 'undefined') {
          localStorage.removeItem('authToken');
        }
        set({ user: null, token: null, isAuthenticated: false, isLoading: false });
      },

      updateUser: (updates) =>
        set((state) => ({
          user: state.user ? { ...state.user, ...updates } : null,
        })),

      setLoading: (loading) => set({ isLoading: loading }),
    }),
    {
      name: 'woof-auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
