import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { petsApi } from './pets';

describe('petsApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
  });

  it('reads only the authenticated user owned-pet collection', async () => {
    transport.get.mockResolvedValue({ pets: [], total: 0, skip: 0, take: 100 });

    await petsApi.getMine();

    expect(transport.get).toHaveBeenCalledWith('/pets/me', { params: { take: 100 } });
  });

  it('creates a dog without inventing temperament or health fields', async () => {
    transport.post.mockResolvedValue({ id: 'pet-1', name: 'Mochi', species: 'DOG' });

    await petsApi.createDog({
      name: 'Mochi',
      species: 'DOG',
      breed: 'Mix',
      sex: 'UNKNOWN',
    });

    expect(transport.post).toHaveBeenCalledWith('/pets', {
      name: 'Mochi',
      species: 'DOG',
      breed: 'Mix',
      sex: 'UNKNOWN',
    });
    const payload = transport.post.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(payload).not.toHaveProperty('temperament');
    expect(payload).not.toHaveProperty('vaccinations');
  });
});
