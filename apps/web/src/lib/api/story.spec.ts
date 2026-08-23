import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  put: vi.fn(),
}));

vi.mock('./client', () => ({
  apiClient: transport,
}));

import { storyApi } from './story';

describe('storyApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.put.mockReset();
  });

  it('reads the unified story with bounded pagination parameters', async () => {
    transport.get.mockResolvedValue({ moments: [] });

    await storyApi.get({ petId: 'pet-1', before: '2026-08-20T12:00:00.000Z', limit: 40 });

    expect(transport.get).toHaveBeenCalledWith('/story', {
      params: {
        petId: 'pet-1',
        before: '2026-08-20T12:00:00.000Z',
        limit: 40,
      },
    });
  });

  it('does not invent query parameters for the default story read', async () => {
    transport.get.mockResolvedValue({ moments: [] });

    await storyApi.get();

    expect(transport.get).toHaveBeenCalledWith('/story', { params: undefined });
  });

  it('sends source identity and owner curation only, never copied source truth', async () => {
    transport.put.mockResolvedValue({ state: 'SAVED' });

    await storyApi.curate({
      sourceType: 'ACTIVITY',
      sourceId: 'activity-1',
      action: 'SAVE',
      note: 'The sunset loop.',
    });

    expect(transport.put).toHaveBeenCalledWith('/story/curation', {
      sourceType: 'ACTIVITY',
      sourceId: 'activity-1',
      action: 'SAVE',
      note: 'The sunset loop.',
    });
    const body = transport.put.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(body).not.toHaveProperty('title');
    expect(body).not.toHaveProperty('summary');
    expect(body).not.toHaveProperty('occurredAt');
    expect(body).not.toHaveProperty('metrics');
  });

  it('clears curation by reference instead of deleting the source', async () => {
    transport.put.mockResolvedValue({ state: null });

    await storyApi.curate({
      sourceType: 'MEDIA',
      sourceId: 'media-1',
      action: 'CLEAR',
    });

    expect(transport.put).toHaveBeenCalledWith('/story/curation', {
      sourceType: 'MEDIA',
      sourceId: 'media-1',
      action: 'CLEAR',
    });
  });
});
