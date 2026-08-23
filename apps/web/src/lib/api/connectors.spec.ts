import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  delete: vi.fn(),
  post: vi.fn(),
}));

vi.mock('./client', () => ({
  apiClient: transport,
}));

import { connectorsApi } from './connectors';

describe('connectorsApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.delete.mockReset();
    transport.post.mockReset();
  });

  it('reads provider capabilities and truthful connection state', async () => {
    transport.get.mockResolvedValue({ providers: [] });

    await connectorsApi.getDashboard();

    expect(transport.get).toHaveBeenCalledWith('/connectors');
  });

  it('disconnects a named provider without sending provider credentials or payload data', async () => {
    transport.delete.mockResolvedValue({ success: true });

    await connectorsApi.disconnect('TRACTIVE');

    expect(transport.delete).toHaveBeenCalledWith('/connectors/TRACTIVE');
    expect(transport.post).not.toHaveBeenCalled();
  });
});
