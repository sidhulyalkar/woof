import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('./client', () => ({
  apiClient: transport,
}));

import { adventureApi, type CompleteQuestInput } from './adventure';

describe('adventureApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
    window.localStorage.clear();
    window.history.replaceState({}, '', '/');
  });

  it('reads the server-owned dashboard with an explicit pet scope', async () => {
    transport.get.mockResolvedValue({ quests: [] });

    await adventureApi.getMine('pet-1');

    expect(transport.get).toHaveBeenCalledWith('/adventure/me', {
      params: { petId: 'pet-1' },
    });
  });

  it('uses the active dog context when the caller does not override it', async () => {
    transport.get.mockResolvedValue({ quests: [] });
    window.history.replaceState({}, '', '/?pet=pet-2');

    await adventureApi.getMine();

    expect(transport.get).toHaveBeenCalledWith('/adventure/me', {
      params: { petId: 'pet-2' },
    });
  });

  it('selects a quest using only quest identity and pet ownership context', async () => {
    transport.post.mockResolvedValue({ ok: true });

    await adventureApi.selectQuest('quest-1', 'pet-1');

    expect(transport.post).toHaveBeenCalledWith('/adventure/quests/quest-1/select', {
      petId: 'pet-1',
    });
  });

  it('never adds client-controlled XP to quest completion payloads', async () => {
    transport.post.mockResolvedValue({
      reward: {
        careEventId: 'event-1',
        ledgerId: 'ledger-1',
        bondXp: 12,
        pathway: 'BOND',
        policyVersion: 'bond-xp-v1',
        explanation: 'Server decision',
        duplicate: false,
      },
      message: 'Done',
    });

    const input: CompleteQuestInput = {
      petId: 'pet-1',
      dogExperience: 'comfortable',
      ownerExperience: 'fine',
      safeOptOut: true,
      note: 'Stopped when the dog asked for space.',
    };

    await adventureApi.completeQuest('quest-1', input);

    expect(transport.post).toHaveBeenCalledWith('/adventure/quests/quest-1/complete', input);
    const sentBody = transport.post.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(sentBody).not.toHaveProperty('xp');
    expect(sentBody).not.toHaveProperty('bondXp');
    expect(sentBody).not.toHaveProperty('pathwayXp');
  });
});
