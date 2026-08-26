import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  put: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { petsApi } from './pets';

describe('petsApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
    transport.put.mockReset();
  });

  it('reads only the authenticated user owned-pet collection', async () => {
    transport.get.mockResolvedValue({ pets: [], total: 0, skip: 0, take: 100 });

    await petsApi.getMine();

    expect(transport.get).toHaveBeenCalledWith('/pets/me', { params: { take: 100 } });
  });

  it('creates a dog without inventing temperament or health fields', async () => {
    transport.post.mockResolvedValue({
      id: 'pet-1',
      name: 'Mochi',
      species: 'DOG',
      householdMemberships: [{ householdId: 'house-1' }],
    });

    await petsApi.createDog({
      name: 'Mochi',
      species: 'DOG',
      breed: 'Mix',
      sex: 'UNKNOWN',
      creationKey: 'first-adventure:abc',
    });

    expect(transport.post).toHaveBeenCalledWith('/pets', {
      name: 'Mochi',
      species: 'DOG',
      breed: 'Mix',
      sex: 'UNKNOWN',
      creationKey: 'first-adventure:abc',
    });
    const payload = transport.post.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(payload).not.toHaveProperty('temperament');
    expect(payload).not.toHaveProperty('vaccinations');
    expect(payload).not.toHaveProperty('avatarUrl');
  });

  it('creates a generic pet and preserves the pair household returned by the server', async () => {
    transport.post.mockResolvedValue({
      id: 'pet-2',
      name: 'Pip',
      species: 'OTHER',
      householdMemberships: [{ householdId: 'house-2' }],
    });

    const pet = await petsApi.createPet({
      name: 'Pip',
      species: 'OTHER',
      birthdate: '2023-05-01',
      creationKey: 'first-adventure:def',
    });

    expect(transport.post).toHaveBeenCalledWith('/pets', {
      name: 'Pip',
      species: 'OTHER',
      birthdate: '2023-05-01',
      creationKey: 'first-adventure:def',
    });
    expect(pet.householdMemberships[0]?.householdId).toBe('house-2');
  });

  it('updates an existing pet instead of requiring a second create during recovery', async () => {
    transport.put.mockResolvedValue({
      id: 'pet/one',
      name: 'Mochi',
      species: 'DOG',
    });

    await petsApi.updatePet('pet/one', { avatarUrl: 'https://cdn.example.test/mochi.jpg' });

    expect(transport.put).toHaveBeenCalledWith('/pets/pet%2Fone', {
      avatarUrl: 'https://cdn.example.test/mochi.jpg',
    });
  });
});
