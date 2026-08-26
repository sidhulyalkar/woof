import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { adaptiveProfileApi } from './adaptive-profile';

describe('adaptiveProfileApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
  });

  it('encodes household and pet identities in the pair-scoped profile path', async () => {
    transport.get.mockResolvedValue({
      schemaVersion: 'adaptive-profile-v1',
      householdId: 'house hold',
      petId: 'pet/one',
      dimensions: [],
      coverage: { known: [], learning: [], unknown: [] },
    });

    await adaptiveProfileApi.getState('house hold', 'pet/one');

    expect(transport.get).toHaveBeenCalledWith('/adventure/profile/house%20hold/pet%2Fone');
  });

  it('passes the bounded question response through without inventing reward fields', async () => {
    const input = {
      responseId: 'first-adventure-v1:pet-1:profile-owner-goals-v1',
      questionId: 'profile-owner-goals-v1',
      outcome: 'ANSWERED' as const,
      answers: ['TRAINING'],
    };
    transport.post.mockResolvedValue({ duplicate: false, profile: {} });

    await adaptiveProfileApi.recordQuestionResponse('house-1', 'pet-1', input);

    expect(transport.post).toHaveBeenCalledWith(
      '/adventure/profile/house-1/pet-1/questions/respond',
      input
    );
    const payload = transport.post.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(payload).not.toHaveProperty('bondXp');
    expect(payload).not.toHaveProperty('totalPoints');
    expect(payload).not.toHaveProperty('rewardLedger');
  });
});
