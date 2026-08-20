import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface AuthPet {
  id: string;
  name: string;
  species: string;
  breed?: string | null;
  sex?: string | null;
  birthdate?: string | null;
  temperament?: unknown;
  avatarUrl?: string | null;
}

export interface AuthUser {
  id: string;
  handle: string;
  email: string;
  bio?: string | null;
  avatarUrl?: string | null;
  visibility?: 'PUBLIC' | 'FRIENDS_ONLY' | 'PRIVATE';
  points?: number;
  totalPoints?: number;
  isVerified?: boolean;
  createdAt?: string;
  pets?: AuthPet[];
  _count?: {
    posts?: number;
    activities?: number;
  };
}

interface AuthState {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  setAuth: (user: AuthUser, token: string) => void;
  logout: () => void;
  updateUser: (user: Partial<AuthUser>) => void;
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
        if (typeof window !== 'undefined') {
          localStorage.setItem('authToken', token);
        }
        set({ user, token, isAuthenticated: true, isLoading: false });
      },

      logout: () => {
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
    },
  ),
);
