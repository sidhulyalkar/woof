import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { meetupProposalsApi } from './meetup-proposals';

describe('meetupProposalsApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
    transport.put.mockReset();
    transport.delete.mockReset();
  });

  it('creates a public-place proposal without client coordinates', async () => {
    const input = {
      recipientId: 'member-2',
      suggestedTime: '2026-08-25T18:00:00.000Z',
      suggestedVenue: { name: 'Neighborhood park', type: 'public_place' },
    };
    transport.post.mockResolvedValue({ id: 'proposal-1' });

    await meetupProposalsApi.create(input);

    expect(transport.post).toHaveBeenCalledWith('/meetup-proposals', input);
    expect(input.suggestedVenue).not.toHaveProperty('lat');
    expect(input.suggestedVenue).not.toHaveProperty('lng');
  });

  it('submits the three structured learning answers plus explicit safety feedback', async () => {
    const outcome = {
      occurred: true,
      dogExperience: 'comfortable' as const,
      ownerExperience: 'great' as const,
      meetAgain: 'yes' as const,
      checklistOk: true,
    };
    transport.put.mockResolvedValue({ feedbackRecorded: true });

    await meetupProposalsApi.complete('proposal-1', outcome);

    expect(transport.put).toHaveBeenCalledWith('/meetup-proposals/proposal-1/complete', outcome);
  });
});
