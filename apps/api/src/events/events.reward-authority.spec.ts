import { BadRequestException } from '@nestjs/common';
import { EventsService } from './events.service';

describe('EventsService reward authority', () => {
  const eventId = 'event-1';
  const userId = 'user-1';

  function buildService() {
    const prisma = {
      eventRSVP: {
        findUnique: jest.fn(),
        update: jest.fn(),
      },
      eventFeedback: {
        findUnique: jest.fn(),
        create: jest.fn(),
        update: jest.fn(),
      },
      pointTransaction: {
        create: jest.fn(),
      },
      user: {
        update: jest.fn(),
      },
    };
    const service = new EventsService(prisma as never);
    return { service, prisma };
  }

  it('checks in without mutating legacy points or advertising rewards', async () => {
    const { service, prisma } = buildService();
    const checkedInAt = new Date('2026-08-28T12:00:00.000Z');

    prisma.eventRSVP.findUnique.mockResolvedValue({
      eventId,
      userId,
      checkedInAt: null,
    });
    prisma.eventRSVP.update.mockResolvedValue({
      eventId,
      userId,
      checkedInAt,
    });

    const result = await service.checkIn(eventId, userId);

    expect(result.message).not.toMatch(/points/i);
    expect(result).not.toHaveProperty('pointsAwarded');
    expect(prisma.pointTransaction.create).not.toHaveBeenCalled();
    expect(prisma.user.update).not.toHaveBeenCalled();
  });

  it('rejects duplicate check-ins without attempting a reward write', async () => {
    const { service, prisma } = buildService();

    prisma.eventRSVP.findUnique.mockResolvedValue({
      eventId,
      userId,
      checkedInAt: new Date('2026-08-28T11:00:00.000Z'),
    });

    await expect(service.checkIn(eventId, userId)).rejects.toBeInstanceOf(BadRequestException);
    expect(prisma.eventRSVP.update).not.toHaveBeenCalled();
    expect(prisma.pointTransaction.create).not.toHaveBeenCalled();
  });

  it('records first feedback without legacy point awards', async () => {
    const { service, prisma } = buildService();
    const feedback = {
      eventId,
      userId,
      vibeScore: 4,
      petDensity: 'just_right',
      surfaceType: 'grass',
      crowding: 'moderate',
      noiseLevel: 'quiet',
      tags: ['friendly'],
      notes: 'Great turnout',
    };

    prisma.eventRSVP.findUnique.mockResolvedValue({ eventId, userId });
    prisma.eventFeedback.findUnique.mockResolvedValue(null);
    prisma.eventFeedback.create.mockResolvedValue(feedback);

    const result = await service.submitFeedback(eventId, userId, {
      vibeScore: 4,
      petDensity: 'just_right',
      surfaceType: 'grass',
      crowding: 'moderate',
      noiseLevel: 'quiet',
      tags: ['friendly'],
      notes: 'Great turnout',
    });

    expect(result.message).not.toMatch(/points/i);
    expect(result).not.toHaveProperty('pointsAwarded');
    expect(prisma.pointTransaction.create).not.toHaveBeenCalled();
    expect(prisma.user.update).not.toHaveBeenCalled();
  });

  it('updates existing feedback without legacy point awards', async () => {
    const { service, prisma } = buildService();
    const existing = {
      eventId,
      userId,
      vibeScore: 3,
      petDensity: 'just_right',
      surfaceType: 'grass',
      crowding: 'moderate',
      noiseLevel: 'quiet',
      tags: [],
      notes: null,
    };

    prisma.eventRSVP.findUnique.mockResolvedValue({ eventId, userId });
    prisma.eventFeedback.findUnique.mockResolvedValue(existing);
    prisma.eventFeedback.update.mockResolvedValue({
      ...existing,
      vibeScore: 5,
      notes: 'Even better on the second visit',
    });

    const result = await service.submitFeedback(eventId, userId, {
      vibeScore: 5,
      notes: 'Even better on the second visit',
    });

    expect(result.message).toBe('Feedback updated successfully.');
    expect(result).not.toHaveProperty('pointsAwarded');
    expect(prisma.eventFeedback.create).not.toHaveBeenCalled();
    expect(prisma.pointTransaction.create).not.toHaveBeenCalled();
  });
});
