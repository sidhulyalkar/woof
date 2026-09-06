import * as SecureStore from 'expo-secure-store';

const REGISTRATION_RECOVERY_KEY = 'woof:native-onboarding:registration:v1';
const PET_CREATION_RECOVERY_KEY = 'woof:native-onboarding:pet-create:v1';
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

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
  ambiguous?: boolean;
};

function registrationReplayKey() {
  // RegisterDto requires a UUID. This value is only an idempotency identity,
  // never an authentication credential, so native does not need to add a new
  // cryptography dependency solely for replay-key generation.
  let entropy = Date.now();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (slot) => {
    const nibble = ((entropy + Math.random() * 16) % 16) | 0;
    entropy = Math.floor(entropy / 16);
    const value = slot === 'x' ? nibble : (nibble & 0x3) | 0x8;
    return value.toString(16);
  });
}

function replayKey(prefix: string) {
  // Pet creation accepts an opaque bounded string. This remains separate from
  // the stricter UUID-shaped registration replay contract.
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
    UUID_V4.test(current.registrationKey) &&
    current.email === canonicalEmail &&
    current.handle === canonicalHandle
  ) {
    return current;
  }

  const next: RegistrationRecovery = {
    registrationKey: registrationReplayKey(),
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
    ambiguous: false,
  };
  await SecureStore.setItemAsync(PET_CREATION_RECOVERY_KEY, JSON.stringify(next));
  return next;
}

export async function markPetCreationAmbiguous(ambiguous: boolean) {
  const current = await readJson<PetCreationRecovery>(PET_CREATION_RECOVERY_KEY);
  if (!current) return;
  await SecureStore.setItemAsync(
    PET_CREATION_RECOVERY_KEY,
    JSON.stringify({ ...current, ambiguous })
  );
}

export async function hasAmbiguousPetCreationRecovery() {
  const current = await readJson<PetCreationRecovery>(PET_CREATION_RECOVERY_KEY);
  return current?.ambiguous === true;
}

export async function readPetCreationRecovery(ownerId: string) {
  const current = await readJson<PetCreationRecovery>(PET_CREATION_RECOVERY_KEY);
  return current?.ownerId === ownerId ? current : null;
}

export async function clearPetCreationRecovery() {
  await SecureStore.deleteItemAsync(PET_CREATION_RECOVERY_KEY);
}
