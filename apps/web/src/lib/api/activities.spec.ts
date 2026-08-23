import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { activitiesApi } from './activities';

describe('activitiesApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
  });

  it('reads paginated canonical history scoped to the active dog', async () => {
    transport.get.mockResolvedValue({ activities: [], total: 0, skip: 20, take: 20 });

    await activitiesApi.getMine({ petId: 'pet-2', skip: 20, take: 20 });

    expect(transport.get).toHaveBeenCalledWith('/activities', {
      params: { petId: 'pet-2', skip: 20, take: 20 },
    });
  });

  it('writes a completed activity without client-controlled rewards or fake route data', async () => {
    transport.post.mockResolvedValue({ id: 'activity-1' });
    const input = {
      petIds: ['pet-2'],
      type: 'WALK',
      startedAt: '2026-08-22T17:00:00.000Z',
      endedAt: '2026-08-22T17:30:00.000Z',
      jointMetrics: {
        source: 'MANUAL_QUICK_LOG',
        enteredDurationMinutes: 30,
      },
    };

    await activitiesApi.create(input);

    expect(transport.post).toHaveBeenCalledWith('/activities', input);
    const payload = transport.post.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(payload).not.toHaveProperty('route');
    expect(payload).not.toHaveProperty('distance');
    expect(payload).not.toHaveProperty('bondXp');
    expect(payload).not.toHaveProperty('xp');
  });
});
