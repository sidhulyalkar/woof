import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  delete: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { behaviorVisionApi } from './behavior-vision';

describe('behaviorVisionApi shadow transport', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
    transport.delete.mockReset();
  });

  it('reads a pet-scoped shadow snapshot without a client authority flag', async () => {
    transport.get.mockResolvedValue({
      policy: {
        mode: 'shadow-evidence-only',
        canInfluenceCompatibility: false,
      },
      evaluation: {},
      moments: [],
    });

    await behaviorVisionApi.shadow('pet-1');

    expect(transport.get).toHaveBeenCalledWith('/behavior-vision/shadow', {
      params: { petId: 'pet-1' },
    });
  });
});
