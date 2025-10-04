import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: string;
  email: string;
  username: string;
  firstName?: string;
  lastName?: string;
  avatar?: string;
}

interface Pet {
  id: string;
  name: string;
  type: string;
  breed?: string;
  avatar?: string;
  age?: number;
}

interface SessionState {
  user: User | null;
  pets: Pet[];
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;

  // Actions
  setUser: (user: User | null) => void;
  setPets: (pets: Pet[]) => void;
  setTokens: (accessToken: string, refreshToken: string) => void;
  login: (user: User, accessToken: string, refreshToken: string) => void;
  logout: () => void;
  updateUser: (updates: Partial<User>) => void;
  addPet: (pet: Pet) => void;
  updatePet: (petId: string, updates: Partial<Pet>) => void;
  removePet: (petId: string) => void;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      user: null,
      pets: [],
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,

      setUser: (user) => set({ user, isAuthenticated: !!user }),

      setPets: (pets) => set({ pets }),

      setTokens: (accessToken, refreshToken) =>
        set({ accessToken, refreshToken }),

      login: (user, accessToken, refreshToken) =>
        set({
          user,
          accessToken,
          refreshToken,
          isAuthenticated: true
        }),

      logout: () =>
        set({
          user: null,
          pets: [],
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false
        }),

      updateUser: (updates) =>
        set((state) => ({
          user: state.user ? { ...state.user, ...updates } : null
        })),

      addPet: (pet) =>
        set((state) => ({
          pets: [...state.pets, pet]
        })),

      updatePet: (petId, updates) =>
        set((state) => ({
          pets: state.pets.map((pet) =>
            pet.id === petId ? { ...pet, ...updates } : pet
          )
        })),

      removePet: (petId) =>
        set((state) => ({
          pets: state.pets.filter((pet) => pet.id !== petId)
        }))
    }),
    {
      name: 'woof-session',
      partialize: (state) => ({
        user: state.user,
        pets: state.pets,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
