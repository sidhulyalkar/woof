import { BadRequestException } from '@nestjs/common';
import { EventsService } from './events.service';

const eventId = 'event-1';
const userId = 'user-1';

function buildService() {
  const prisma = {
    eventRSVP: {
      updateMany: jest.fn(),
      findUnique: jest.fn(),
    },
    eventFeedback: {
      upsert: jest.fn(),
    },
  };
  return {
    prisma,
    service: new EventsService(prisma as never),
  };
}

describe('EventsService community reward authority', () => {
  it('claims an unchecked RSVP with one conditional transition and no reward response', async () => {
    const { prisma, service } = buildService();
    const canonical = {
      eventId,
      userId,
      status: 'going',
      checkedInAt: new Date('2026-08-31T12:00:00.000Z'),
    };
    prisma.eventRSVP.updateMany.mockResolvedValue({ count: 1 });
    prisma.eventRSVP.findUnique.mockResolvedValue(canonical);

    const result = await service.checkIn(eventId, userId);

    expect(prisma.eventRSVP.updateMany).toHaveBeenCalledWith({
      where: { eventId, userId, checkedInAt: null },
      data: { checkedInAt: expect.any(Date) },
    });
    expect(result).toEqual({
      ...canonical,
      message: 'Checked in successfully. Thanks for joining the community event.',
    });
    expect(result).not.toHaveProperty('pointsAwarded');
    expect(result.message).not.toMatch(/earned|points?/i);
  });

  it('settles a repeated or concurrent-loser check-in as an acknowledged no-op', async () => {
    const { prisma, service } = buildService();
    const canonical = {
      eventId,
      userId,
      status: 'going',
      checkedInAt: new Date('2026-08-31T12:00:00.000Z'),
    };
    prisma.eventRSVP.updateMany.mockResolvedValue({ count: 0 });
    prisma.eventRSVP.findUnique.mockResolvedValue(canonical);

    const result = await service.checkIn(eventId, userId);

    expect(result).toEqual({
      ...canonical,
      message: 'Already checked in. Attendance is unchanged.',
    });
    expect(result).not.toHaveProperty('pointsAwarded');
  });

  it('distinguishes a missing RSVP from an already-completed retry', async () => {
    const { prisma, service } = buildService();
    prisma.eventRSVP.updateMany.mockResolvedValue({ count: 0 });
    prisma.eventRSVP.findUnique.mockResolvedValue(null);

    await expect(service.checkIn(eventId, userId)).rejects.toBeInstanceOf(BadRequestException);
  });

  it('uses one composite-key upsert for feedback and exposes no reward semantics', async () => {
    const { prisma, service } = buildService();
    prisma.eventRSVP.findUnique.mockResolvedValue({ eventId, userId });
    prisma.eventFeedback.upsert.mockResolvedValue({
      eventId,
      userId,
      vibeScore: 5,
      tags: ['friendly'],
      notes: 'Good meetup',
    });

    const result = await service.submitFeedback(eventId, userId, {
      vibeScore: 5,
      tags: ['friendly'],
      notes: 'Good meetup',
    });

    expect(prisma.eventFeedback.upsert).toHaveBeenCalledWith({
      where: { eventId_userId: { eventId, userId } },
      create: expect.objectContaining({
        eventId,
        userId,
        vibeScore: 5,
        tags: ['friendly'],
      }),
      update: expect.objectContaining({
        vibeScore: 5,
        tags: ['friendly'],
      }),
    });
    expect(result).not.toHaveProperty('pointsAwarded');
    expect(result.message).toBe(
      'Feedback saved. Thanks for helping the community learn about this event.'
    );
  });

  it('preserves feedback replacement semantics when optional tags are omitted', async () => {
    const { prisma, service } = buildService();
    prisma.eventRSVP.findUnique.mockResolvedValue({ eventId, userId });
    prisma.eventFeedback.upsert.mockResolvedValue({ eventId, userId, vibeScore: 4, tags: [] });

    await service.submitFeedback(eventId, userId, { vibeScore: 4 });

    expect(prisma.eventFeedback.upsert).toHaveBeenCalledWith(
      expect.objectContaining({
        create: expect.objectContaining({ tags: [] }),
        update: expect.objectContaining({ tags: [] }),
      })
    );
  });

  it('requires an RSVP before feedback without attempting the feedback transition', async () => {
    const { prisma, service } = buildService();
    prisma.eventRSVP.findUnique.mockResolvedValue(null);

    await expect(service.submitFeedback(eventId, userId, { vibeScore: 4 })).rejects.toBeInstanceOf(
      BadRequestException
    );
    expect(prisma.eventFeedback.upsert).not.toHaveBeenCalled();
  });
});
