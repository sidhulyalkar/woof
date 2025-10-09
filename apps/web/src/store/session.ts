import { create } from 'zustand';

interface User {
  id: string;
  email: string;
  name?: string;
  username?: string;
  avatarUrl?: string;
  pets?: Pet[];
}

interface Pet {
  id: string;
  name: string;
  species: string;
  breed?: string;
  age?: number;
  avatarUrl?: string;
}

interface SessionState {
  user: User | null;
  token: string | null;
  setSession: (user: User, token: string) => void;
  clearSession: () => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  user: null,
  token: null,
  setSession: (user, token) => set({ user, token }),
  clearSession: () => set({ user: null, token: null }),
}));
