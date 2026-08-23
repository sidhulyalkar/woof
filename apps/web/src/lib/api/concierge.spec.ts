import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({ get: vi.fn() }));

vi.mock('./client', () => ({ apiClient: transport }));

import { conciergeApi } from './concierge';

describe('conciergeApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.get.mockResolvedValue({ suggestions: [] });
  });

  it('reads today without inventing a pet selector', async () => {
    await conciergeApi.getToday();

    expect(transport.get).toHaveBeenCalledWith('/concierge/today', { params: undefined });
  });

  it('passes an explicit pet selection only when supplied', async () => {
    await conciergeApi.getToday('pet-1');

    expect(transport.get).toHaveBeenCalledWith('/concierge/today', {
      params: { petId: 'pet-1' },
    });
  });

  it('exposes no mutation method from the Concierge web client', () => {
    expect(Object.keys(conciergeApi)).toEqual(['getToday']);
  });
});
