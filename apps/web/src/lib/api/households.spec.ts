import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { householdsApi } from './households';

describe('householdsApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
  });

  it('loads only households authorized for the current user', async () => {
    transport.get.mockResolvedValue([]);

    await householdsApi.getMine();

    expect(transport.get).toHaveBeenCalledWith('/households/me');
  });
});
