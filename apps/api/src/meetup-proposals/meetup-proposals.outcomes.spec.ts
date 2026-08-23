import { MeetupProposalsService } from './meetup-proposals.service';

describe('MeetupProposalsService structured outcomes', () => {
  it('stores tiny structured answers in canonical telemetry and normalized proposal tags', async () => {
    const proposal = {
      id: 'proposal-1',
      proposerId: 'user-1',
      recipientId: 'user-2',
      status: 'accepted',
      rating: null,
      feedbackTags: [],
      checklistOk: true,
      occurredAt: null,
      notes: null,
    };
    const prisma = {
      meetupProposal: {
        findUnique: jest.fn().mockResolvedValue(proposal),
        update: jest.fn().mockImplementation(async ({ data }: { data: Record<string, unknown> }) => ({
          ...proposal,
          ...data,
        })),
      },
      telemetry: {
        findFirst: jest.fn().mockResolvedValue(null),
        create: jest.fn().mockResolvedValue({ id: 'telemetry-1' }),
      },
    };
    const service = new MeetupProposalsService(prisma as never);

    await service.complete('proposal-1', 'user-1', {
      occurred: true,
      dogExperience: 'comfortable' as never,
      ownerExperience: 'great' as never,
      meetAgain: 'yes' as never,
      checklistOk: true,
    });

    expect(prisma.telemetry.create).toHaveBeenCalledWith({
      data: {
        userId: 'user-1',
        source: 'meetup',
        event: 'MEETUP_OUTCOME_REPORTED',
        data: expect.objectContaining({
          proposalId: 'proposal-1',
          otherUserId: 'user-2',
          dogExperience: 'comfortable',
          ownerExperience: 'great',
          meetAgain: 'yes',
          feedbackTags: ['dog_comfortable', 'owner_great', 'meet_again_yes'],
          checklistOk: true,
        }),
      },
    });
    expect(prisma.meetupProposal.update).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({
          status: 'completed',
          feedbackTags: ['dog_comfortable', 'owner_great', 'meet_again_yes'],
        }),
      }),
    );
  });
});
