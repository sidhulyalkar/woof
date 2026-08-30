'use client';

import { useMemo } from 'react';
import { useAuthStore, type AuthPet, type AuthUser } from '@/lib/stores/auth-store';

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

export interface SessionViewState {
  user: SessionUser | null;
  pets: SessionPet[];
}

function ageFromBirthdate(birthdate?: string | null) {
  if (!birthdate) return undefined;
  const born = new Date(birthdate);
  if (Number.isNaN(born.getTime())) return undefined;

  const now = new Date();
  let age = now.getFullYear() - born.getFullYear();
  const beforeBirthday =
    now.getMonth() < born.getMonth() ||
    (now.getMonth() === born.getMonth() && now.getDate() < born.getDate());
  if (beforeBirthday) age -= 1;
  return Math.max(0, age);
}

function projectPet(pet: AuthPet): SessionPet {
  return {
    id: pet.id,
    name: pet.name,
    species: pet.species,
    breed: pet.breed ?? undefined,
    age: ageFromBirthdate(pet.birthdate),
    avatar: pet.avatarUrl ?? undefined,
    avatarUrl: pet.avatarUrl ?? undefined,
    bio: pet.bio ?? undefined,
  };
}

function projectUser(user: AuthUser | null): SessionUser | null {
  if (!user) return null;
  const pets = user.pets?.map(projectPet);

  return {
    id: user.id,
    email: user.email,
    username: user.handle,
    handle: user.handle,
    avatar: user.avatarUrl ?? undefined,
    avatarUrl: user.avatarUrl ?? undefined,
    bio: user.bio ?? undefined,
    location: user.location ?? undefined,
    createdAt: user.createdAt,
    points: user.points ?? user.totalPoints ?? 0,
    totalPoints: user.totalPoints,
    isVerified: user.isVerified,
    pets,
  };
}

/**
 * Compatibility-only presentation projection for older UI surfaces.
 *
 * This module owns no credentials, persistence, authenticated state, logout,
 * refresh lifecycle, or server authority. All values derive synchronously from
 * the canonical `useAuthStore`; callers should migrate to that store directly.
 */
export function useSessionStore(): SessionViewState;
export function useSessionStore<T>(selector: (state: SessionViewState) => T): T;
export function useSessionStore<T>(selector?: (state: SessionViewState) => T) {
  const authUser = useAuthStore((state) => state.user);
  const view = useMemo<SessionViewState>(() => {
    const user = projectUser(authUser);
    return { user, pets: user?.pets ?? [] };
  }, [authUser]);

  return selector ? selector(view) : view;
}
