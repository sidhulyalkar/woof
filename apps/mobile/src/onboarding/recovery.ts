import * as SecureStore from 'expo-secure-store';

const REGISTRATION_RECOVERY_KEY = 'woof:native-onboarding:registration:v1';
const PET_CREATION_RECOVERY_KEY = 'woof:native-onboarding:pet-create:v1';

type RegistrationRecovery = {
  registrationKey: string;
  email: string;
  handle: string;
};

export type PetCreationRecovery = {
  creationKey: string;
  ownerId: string;
  name: string;
  breed?: string;
};

function replayKey(prefix: string) {
  // This is an idempotency identity, not an authentication credential. Two
  // independent random components plus the device clock keep collisions
  // negligible without adding another native dependency to the release graph.
  const random = () => Math.random().toString(36).slice(2, 14);
  return `${prefix}:${Date.now().toString(36)}:${random()}:${random()}`;
}

async function readJson<T>(key: string): Promise<T | null> {
  try {
    const stored = await SecureStore.getItemAsync(key);
    if (!stored) return null;
    return JSON.parse(stored) as T;
  } catch {
    return null;
  }
}

export async function getOrCreateRegistrationRecovery(
  email: string,
  handle: string
): Promise<RegistrationRecovery> {
  const canonicalEmail = email.trim().toLowerCase();
  const canonicalHandle = handle.trim().toLowerCase();
  const current = await readJson<RegistrationRecovery>(REGISTRATION_RECOVERY_KEY);

  if (
    current?.registrationKey &&
    current.email === canonicalEmail &&
    current.handle === canonicalHandle
  ) {
    return current;
  }

  const next: RegistrationRecovery = {
    registrationKey: replayKey('native-register-v1'),
    email: canonicalEmail,
    handle: canonicalHandle,
  };
  await SecureStore.setItemAsync(REGISTRATION_RECOVERY_KEY, JSON.stringify(next));
  return next;
}

export async function clearRegistrationRecovery() {
  await SecureStore.deleteItemAsync(REGISTRATION_RECOVERY_KEY);
}

export async function getOrCreatePetCreationRecovery(
  ownerId: string,
  name: string,
  breed?: string
): Promise<PetCreationRecovery> {
  const canonicalName = name.trim();
  const canonicalBreed = breed?.trim() || undefined;
  const current = await readJson<PetCreationRecovery>(PET_CREATION_RECOVERY_KEY);

  if (
    current?.creationKey &&
    current.ownerId === ownerId &&
    current.name === canonicalName &&
    current.breed === canonicalBreed
  ) {
    return current;
  }

  const next: PetCreationRecovery = {
    creationKey: replayKey('native-first-adventure-v1'),
    ownerId,
    name: canonicalName,
    breed: canonicalBreed,
  };
  await SecureStore.setItemAsync(PET_CREATION_RECOVERY_KEY, JSON.stringify(next));
  return next;
}

export async function readPetCreationRecovery(ownerId: string) {
  const current = await readJson<PetCreationRecovery>(PET_CREATION_RECOVERY_KEY);
  return current?.ownerId === ownerId ? current : null;
}

export async function clearPetCreationRecovery() {
  await SecureStore.deleteItemAsync(PET_CREATION_RECOVERY_KEY);
}
