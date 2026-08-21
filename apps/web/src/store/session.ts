import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface SessionPet {
  id: string;
  name: string;
  species: string;
  breed?: string;
  age?: number;
  avatar?: string;
  avatarUrl?: string;
  bio?: string;
}

export interface SessionUser {
  id: string;
  email: string;
  name?: string;
  username?: string;
  handle?: string;
  avatar?: string;
  avatarUrl?: string;
  bio?: string;
  location?: string;
  createdAt?: string;
  points?: number;
  totalPoints?: number;
  isVerified?: boolean;
  isAdmin?: boolean;
  pets?: SessionPet[];
}

interface SessionState {
  user: SessionUser | null;
  pets: SessionPet[];
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  login: (user: SessionUser, token: string, refreshToken?: string) => void;
  logout: () => void;
  setSession: (user: SessionUser, token: string) => void;
  clearSession: () => void;
  refreshSession: () => Promise<void>;
}

function normalizePet(pet: SessionPet): SessionPet {
  return {
    ...pet,
    avatar: pet.avatar ?? pet.avatarUrl,
  };
}

function normalizeUser(user: SessionUser): SessionUser {
  const pets = user.pets?.map(normalizePet);
  return {
    ...user,
    username: user.username ?? user.handle ?? user.name,
    avatar: user.avatar ?? user.avatarUrl,
    points: user.points ?? user.totalPoints ?? 0,
    pets,
  };
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => {
      const applySession = (user: SessionUser, token: string, refreshToken?: string) => {
        const normalized = normalizeUser(user);
        if (typeof window !== 'undefined') {
          localStorage.setItem('authToken', token);
        }
        set({
          user: normalized,
          pets: normalized.pets ?? [],
          token,
          refreshToken: refreshToken ?? get().refreshToken,
          isAuthenticated: true,
        });
      };

      const clear = () => {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('authToken');
        }
        set({
          user: null,
          pets: [],
          token: null,
          refreshToken: null,
          isAuthenticated: false,
        });
      };

      return {
        user: null,
        pets: [],
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        login: applySession,
        logout: clear,
        setSession: (user, token) => applySession(user, token),
        clearSession: clear,
        refreshSession: async () => {
          const token = get().token;
          const apiBase = process.env.NEXT_PUBLIC_API_URL;
          if (!token || !apiBase) return;

          const response = await fetch(`${apiBase}/auth/me`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (response.status === 401) {
            clear();
            return;
          }
          if (!response.ok) {
            throw new Error(`Session refresh failed with status ${response.status}`);
          }

          const user = (await response.json()) as SessionUser;
          applySession(user, token);
        },
      };
    },
    {
      name: 'woof-session-storage',
      partialize: (state) => ({
        user: state.user,
        pets: state.pets,
        token: state.token,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
