import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  AUTH_PERSIST_VERSION,
  AUTH_STORAGE_KEY,
  LEGACY_SESSION_STORAGE_KEY,
} from './auth-persist';

export interface AuthPet {
  id: string;
  name: string;
  species: string;
  breed?: string | null;
  sex?: string | null;
  birthdate?: string | null;
  temperament?: unknown;
  avatarUrl?: string | null;
  bio?: string | null;
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
  location?: string | null;
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

function retireLegacySessionStorage() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(LEGACY_SESSION_STORAGE_KEY);
  }
}

// Historical clients may still carry the second `woof-session-storage` Zustand
// snapshot. It is never allowed to hydrate into authority again. Loading the
// canonical store opportunistically retires it, while /auth/me remains the
// server-authoritative recovery path for any still-valid raw bearer session.
retireLegacySessionStorage();

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
        retireLegacySessionStorage();
        set({ user, token, isAuthenticated: true, isLoading: false });
      },

      logout: () => {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('authToken');
        }
        retireLegacySessionStorage();
        set({ user: null, token: null, isAuthenticated: false, isLoading: false });
      },

      updateUser: (updates) =>
        set((state) => ({
          user: state.user ? { ...state.user, ...updates } : null,
        })),

      setLoading: (loading) => set({ isLoading: loading }),
    }),
    {
      name: AUTH_STORAGE_KEY,
      version: AUTH_PERSIST_VERSION,
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
