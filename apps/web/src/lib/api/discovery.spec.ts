import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { discoveryApi } from './discovery';

describe('discoveryApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.put.mockReset();
    transport.delete.mockReset();
  });

  it('sends a precise browser coordinate only to the explicit opt-in endpoint', async () => {
    transport.put.mockResolvedValue({ status: 'OPTED_IN', exactLocationStored: false });

    await discoveryApi.enableLocation(37.7749, -122.4194);

    expect(transport.put).toHaveBeenCalledWith('/discovery/location', {
      latitude: 37.7749,
      longitude: -122.4194,
    });
  });

  it('reads nearby context without asking the client for another coordinate', async () => {
    transport.get.mockResolvedValue({ candidates: [] });

    await discoveryApi.getNearby('pet-1', 5, 20);

    expect(transport.get).toHaveBeenCalledWith('/discovery/nearby/pet-1', {
      params: { radiusKm: 5, limit: 20 },
    });
  });

  it('supports explicit revocation of nearby discovery', async () => {
    transport.delete.mockResolvedValue({ status: 'DISABLED', exactLocationStored: false });

    await discoveryApi.disableLocation();

    expect(transport.delete).toHaveBeenCalledWith('/discovery/location');
  });
});
