import { ConflictException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { MeetupProposalsService } from './meetup-proposals.service';

describe('MeetupProposalsService participant-scoped outcomes', () => {
  function fixture() {
    const proposal = {
      id: 'proposal-1',
      proposerId: 'user-1',
      recipientId: 'user-2',
      status: 'accepted',
      rating: null,
      feedbackTags: [],
      checklistOk: false,
      occurredAt: null,
      notes: 'shared planning note',
    };
    const outcome = {
      id: 'outcome-1',
      proposalId: 'proposal-1',
      participantId: 'user-1',
      occurred: true,
      dogExperience: 'comfortable',
      ownerExperience: 'great',
      meetAgain: 'yes',
      rating: null,
      feedbackTags: ['dog_comfortable', 'meet_again_yes', 'owner_great'],
      checklistOk: true,
      notes: 'private reflection',
      createdAt: new Date(),
    };
    const tx = {
      meetupOutcome: { create: jest.fn().mockResolvedValue(outcome) },
      meetupProposal: {
        updateMany: jest.fn().mockResolvedValue({ count: 1 }),
        findUnique: jest.fn().mockResolvedValue({ ...proposal, status: 'completed' }),
      },
    };
    const prisma = {
      meetupProposal: {
        findUnique: jest.fn().mockResolvedValue(proposal),
      },
      meetupOutcome: {
        findUnique: jest.fn().mockResolvedValue(null),
        findMany: jest.fn().mockResolvedValue([]),
      },
      telemetry: { create: jest.fn().mockResolvedValue({ id: 'telemetry-1' }) },
      $transaction: jest.fn().mockImplementation(async (fn: (value: typeof tx) => unknown) => fn(tx)),
    };
    return { proposal, outcome, prisma, tx };
  }

  it('stores participant-private outcome data without merging it onto the shared proposal', async () => {
    const { prisma, tx } = fixture();
    const service = new MeetupProposalsService(prisma as never);

    const result = await service.complete('proposal-1', 'user-1', {
      occurred: true,
      dogExperience: 'comfortable' as never,
      ownerExperience: 'great' as never,
      meetAgain: 'yes' as never,
      checklistOk: true,
      notes: ' private reflection ',
    });

    expect(tx.meetupOutcome.create).toHaveBeenCalledWith({
      data: expect.objectContaining({
        proposalId: 'proposal-1',
        participantId: 'user-1',
        notes: 'private reflection',
        feedbackTags: ['dog_comfortable', 'meet_again_yes', 'owner_great'],
      }),
    });
    expect(tx.meetupProposal.updateMany).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.not.objectContaining({
          rating: expect.anything(),
          feedbackTags: expect.anything(),
          checklistOk: expect.anything(),
          notes: expect.anything(),
        }),
      })
    );
    expect(result.repeatPlanningEligible).toBe(true);
  });

  it('rejects a divergent retry instead of overwriting the participant outcome', async () => {
    const { prisma, outcome } = fixture();
    const unique = new Prisma.PrismaClientKnownRequestError('unique outcome', {
      code: 'P2002',
      clientVersion: '5.9.1',
    });
    prisma.$transaction.mockRejectedValue(unique);
    prisma.meetupOutcome.findUnique.mockResolvedValue(outcome);
    const service = new MeetupProposalsService(prisma as never);

    await expect(
      service.complete('proposal-1', 'user-1', {
        occurred: true,
        dogExperience: 'not_their_thing' as never,
      })
    ).rejects.toBeInstanceOf(ConflictException);
  });
});
