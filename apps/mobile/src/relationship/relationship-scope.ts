import { useEffect, useSyncExternalStore } from 'react';
import * as SecureStore from 'expo-secure-store';
import { householdsApi, type HouseholdSnapshot } from '../api/households';
import { useAuth } from '../contexts/AuthContext';

const STORAGE_PREFIX = 'woof:selected-relationship-pet:v1:';

export type RelationshipPet = {
  id: string;
  name: string;
  species: string;
  breed?: string | null;
  avatarUrl?: string | null;
  householdNames: string[];
};

type RelationshipSnapshot = {
  userId: string | null;
  pets: RelationshipPet[];
  selectedPetId: string | null;
  loading: boolean;
  error: string | null;
  initialized: boolean;
};

const EMPTY_SNAPSHOT: RelationshipSnapshot = {
  userId: null,
  pets: [],
  selectedPetId: null,
  loading: false,
  error: null,
  initialized: false,
};

let snapshot: RelationshipSnapshot = EMPTY_SNAPSHOT;
let loadPromise: Promise<void> | null = null;
const listeners = new Set<() => void>();

function emit(next: RelationshipSnapshot) {
  snapshot = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function storageKey(userId: string) {
  return `${STORAGE_PREFIX}${userId}`;
}

function flattenAuthorizedPets(households: HouseholdSnapshot[]): RelationshipPet[] {
  const pets = new Map<string, RelationshipPet>();

  for (const household of households) {
    for (const link of household.pets) {
      const existing = pets.get(link.pet.id);
      if (existing) {
        if (!existing.householdNames.includes(household.name)) {
          existing.householdNames.push(household.name);
        }
        continue;
      }

      pets.set(link.pet.id, {
        id: link.pet.id,
        name: link.pet.name,
        species: link.pet.species,
        breed: link.pet.breed,
        avatarUrl: link.pet.avatarUrl,
        householdNames: [household.name],
      });
    }
  }

  return [...pets.values()];
}

async function readStoredPetId(userId: string) {
  try {
    return await SecureStore.getItemAsync(storageKey(userId));
  } catch {
    return null;
  }
}

async function persistSelectedPetId(userId: string, petId: string | null) {
  try {
    if (petId) await SecureStore.setItemAsync(storageKey(userId), petId);
    else await SecureStore.deleteItemAsync(storageKey(userId));
  } catch {
    // Persistence is convenience only. Server household authority remains canonical.
  }
}

async function loadAuthorizedRelationships(userId: string, force = false) {
  if (!force && snapshot.initialized && snapshot.userId === userId) return;
  if (loadPromise && snapshot.userId === userId) return loadPromise;

  const previous = snapshot.userId === userId ? snapshot : EMPTY_SNAPSHOT;
  emit({ ...previous, userId, loading: true, error: null });

  loadPromise = (async () => {
    try {
      const households = await householdsApi.getMine();
      const pets = flattenAuthorizedPets(households);
      const storedPetId = await readStoredPetId(userId);
      const previousPetId = previous.selectedPetId;
      const selectedPetId =
        [storedPetId, previousPetId].find(
          (candidate): candidate is string =>
            Boolean(candidate) && pets.some((pet) => pet.id === candidate)
        ) ?? pets[0]?.id ?? null;

      emit({
        userId,
        pets,
        selectedPetId,
        loading: false,
        error: null,
        initialized: true,
      });

      if (storedPetId !== selectedPetId) {
        void persistSelectedPetId(userId, selectedPetId);
      }
    } catch {
      emit({
        ...previous,
        userId,
        loading: false,
        error:
          'Woof could not verify your dog relationships. Pet-specific views stay closed until household access can be checked.',
        initialized: true,
      });
    } finally {
      loadPromise = null;
    }
  })();

  return loadPromise;
}

export function useRelationshipScope() {
  const { user } = useAuth();
  const userId = user?.id ?? null;
  const current = useSyncExternalStore(subscribe, () => snapshot, () => snapshot);
  const visible = current.userId === userId ? current : EMPTY_SNAPSHOT;

  useEffect(() => {
    if (userId) void loadAuthorizedRelationships(userId);
  }, [userId]);

  const selectPet = (petId: string) => {
    if (!userId || snapshot.userId !== userId) return;
    if (!snapshot.pets.some((pet) => pet.id === petId)) return;
    if (snapshot.selectedPetId === petId) return;

    emit({ ...snapshot, selectedPetId: petId, error: null });
    void persistSelectedPetId(userId, petId);
  };

  return {
    pets: visible.pets,
    selectedPetId: visible.selectedPetId,
    selectedPet: visible.pets.find((pet) => pet.id === visible.selectedPetId) ?? null,
    loading: visible.loading || (Boolean(userId) && !visible.initialized),
    error: visible.error,
    selectPet,
    refresh: () => (userId ? loadAuthorizedRelationships(userId, true) : Promise.resolve()),
  };
}
