import { randomUUID } from 'node:crypto';
import { ConflictException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { MeetupProposalsService } from './meetup-proposals.service';

describe('MeetupProposalsService outcome authority integration', () => {
  const prisma = new PrismaService();
  const service = new MeetupProposalsService(prisma);
  const userIds: string[] = [];
  const proposalIds: string[] = [];

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    if (proposalIds.length > 0) {
      await prisma.meetupOutcome.deleteMany({ where: { proposalId: { in: proposalIds } } });
      await prisma.meetupProposal.deleteMany({ where: { id: { in: proposalIds } } });
    }
    if (userIds.length > 0) {
      await prisma.user.deleteMany({ where: { id: { in: userIds } } });
    }
    await prisma.$disconnect();
  });

  async function createUser(label: string) {
    const suffix = randomUUID().slice(0, 8);
    const user = await prisma.user.create({
      data: {
        handle: `meetup-${label}-${suffix}`,
        email: `meetup-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    userIds.push(user.id);
    return user.id;
  }

  async function createProposal(
    status = 'accepted',
    suggestedTime = new Date(Date.now() - 60 * 60 * 1000)
  ) {
    const proposerId = await createUser('proposer');
    const recipientId = await createUser('recipient');
    const proposal = await prisma.meetupProposal.create({
      data: {
        proposerId,
        recipientId,
        suggestedTime,
        suggestedVenue: { name: 'Public park', type: 'park', area: 'North loop' },
        status,
        notes: 'shared planning note',
      },
    });
    proposalIds.push(proposal.id);
    return { proposal, proposerId, recipientId };
  }

  it('keeps each participant outcome private and independent', async () => {
    const { proposal, proposerId, recipientId } = await createProposal();

    await service.complete(proposal.id, proposerId, {
      occurred: true,
      dogExperience: 'comfortable' as never,
      ownerExperience: 'great' as never,
      meetAgain: 'yes' as never,
      checklistOk: true,
      notes: 'proposer private note',
    });
    await service.complete(proposal.id, recipientId, {
      occurred: true,
      dogExperience: 'loved_it' as never,
      ownerExperience: 'fine' as never,
      meetAgain: 'maybe' as never,
      checklistOk: true,
      notes: 'recipient private note',
    });

    const proposerRead = await service.findAllForUser(proposerId);
    const recipientRead = await service.findAllForUser(recipientId);
    expect(proposerRead.outcomes).toHaveLength(1);
    expect(recipientRead.outcomes).toHaveLength(1);
    expect(proposerRead.outcomes[0]?.notes).toBe('proposer private note');
    expect(recipientRead.outcomes[0]?.notes).toBe('recipient private note');
    expect(proposerRead.sent[0]).not.toHaveProperty('rating');
    expect(proposerRead.sent[0]).not.toHaveProperty('feedbackTags');
    expect(proposerRead.sent[0]).not.toHaveProperty('checklistOk');

    const shared = await prisma.meetupProposal.findUniqueOrThrow({ where: { id: proposal.id } });
    expect(shared.notes).toBe('shared planning note');
    expect(shared.rating).toBeNull();
    expect(shared.feedbackTags).toEqual([]);
  });

  it('converges exact retries and rejects divergent retries', async () => {
    const { proposal, proposerId } = await createProposal();
    const dto = {
      occurred: true,
      dogExperience: 'comfortable' as never,
      ownerExperience: 'fine' as never,
      meetAgain: 'yes' as never,
      checklistOk: true,
    };

    const first = await service.complete(proposal.id, proposerId, dto);
    const retry = await service.complete(proposal.id, proposerId, dto);
    expect(first.idempotentRetry).toBe(false);
    expect(retry.idempotentRetry).toBe(true);
    expect(
      await prisma.meetupOutcome.count({
        where: { proposalId: proposal.id, participantId: proposerId },
      })
    ).toBe(1);

    await expect(
      service.complete(proposal.id, proposerId, { ...dto, meetAgain: 'no' as never })
    ).rejects.toBeInstanceOf(ConflictException);
  });

  it('serializes concurrent same-participant submissions at the database unique key', async () => {
    const { proposal, proposerId } = await createProposal();
    const dto = {
      occurred: true,
      dogExperience: 'comfortable' as never,
      ownerExperience: 'great' as never,
      meetAgain: 'yes' as never,
      checklistOk: true,
    };

    const results = await Promise.all(
      Array.from({ length: 8 }, () => service.complete(proposal.id, proposerId, dto))
    );
    expect(results.filter((result) => result.idempotentRetry)).toHaveLength(7);
    expect(
      await prisma.meetupOutcome.count({
        where: { proposalId: proposal.id, participantId: proposerId },
      })
    ).toBe(1);
  });

  it('does not let one negative occurrence report cancel the shared plan', async () => {
    const { proposal, proposerId } = await createProposal();
    await service.complete(proposal.id, proposerId, { occurred: false });
    const shared = await prisma.meetupProposal.findUniqueOrThrow({ where: { id: proposal.id } });
    expect(shared.status).toBe('accepted');
    expect(shared.occurredAt).toBeNull();
  });

  it('rejects outcome feedback before the suggested meetup time', async () => {
    const future = new Date(Date.now() + 60 * 60 * 1000);
    const { proposal, proposerId } = await createProposal('accepted', future);

    await expect(
      service.complete(proposal.id, proposerId, { occurred: false })
    ).rejects.toThrow('Meetup feedback opens after the suggested meetup time');
  });

  it('converges duplicate acceptance retries at one guarded transition', async () => {
    const future = new Date(Date.now() + 60 * 60 * 1000);
    const { proposal, recipientId } = await createProposal('pending', future);
    const results = await Promise.all([
      service.updateStatus(proposal.id, recipientId, { status: 'accepted' as never }),
      service.updateStatus(proposal.id, recipientId, { status: 'accepted' as never }),
    ]);
    expect(results).toHaveLength(2);
    expect(results.every((result) => result.status === 'accepted')).toBe(true);
    const shared = await prisma.meetupProposal.findUniqueOrThrow({ where: { id: proposal.id } });
    expect(shared.status).toBe('accepted');
  });

  it('never leaves a positive canonical outcome on a proposal that loses a cancel race', async () => {
    const { proposal, proposerId } = await createProposal();
    const results = await Promise.allSettled([
      service.complete(proposal.id, proposerId, {
        occurred: true,
        dogExperience: 'comfortable' as never,
        ownerExperience: 'fine' as never,
        meetAgain: 'maybe' as never,
        checklistOk: true,
      }),
      service.cancel(proposal.id, proposerId),
    ]);

    expect(results.some((result) => result.status === 'fulfilled')).toBe(true);
    const [shared, outcome] = await Promise.all([
      prisma.meetupProposal.findUniqueOrThrow({ where: { id: proposal.id } }),
      prisma.meetupOutcome.findUnique({
        where: {
          proposalId_participantId: {
            proposalId: proposal.id,
            participantId: proposerId,
          },
        },
      }),
    ]);

    if (shared.status === 'cancelled') {
      expect(outcome).toBeNull();
    } else {
      expect(shared.status).toBe('completed');
      expect(outcome?.occurred).toBe(true);
    }
  });
});
