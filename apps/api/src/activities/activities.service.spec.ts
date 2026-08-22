import { BadRequestException } from '@nestjs/common';
import { CareEventsService } from '../care-events/care-events.service';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { ActivitiesService } from './activities.service';

describe('ActivitiesService dogOS participation', () => {
  const userId = '11111111-1111-4111-8111-111111111111';
  const householdId = '22222222-2222-4222-8222-222222222222';
  const petA = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
  const petB = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';

  function createHarness() {
    const prisma = {
      activity: {
        create: jest.fn(),
        findMany: jest.fn(),
        count: jest.fn(),
        findFirst: jest.fn(),
        update: jest.fn(),
        delete: jest.fn(),
      },
    };
    const careEvents = {
      record: jest.fn().mockResolvedValue({}),
    };
    const households = {
      assertPetAccessible: jest.fn().mockResolvedValue({ id: petA }),
      resolveActivityHousehold: jest.fn().mockResolvedValue(householdId),
      householdActivityWhere: jest.fn().mockReturnValue({ userId }),
    };

    return {
      prisma,
      careEvents,
      households,
      service: new ActivitiesService(
        prisma as unknown as PrismaService,
        careEvents as unknown as CareEventsService,
        households as unknown as HouseholdsService
      ),
    };
  }

  it('stores one real activity with two pet participants instead of duplicating the walk', async () => {
    const { service, prisma, households } = createHarness();
    const startedAt = '2026-08-21T17:00:00.000Z';
    const endedAt = '2026-08-21T17:30:00.000Z';

    prisma.activity.create.mockResolvedValue({
      id: 'activity-1',
      userId,
      householdId,
      petId: petA,
      type: 'WALK',
      startedAt: new Date(startedAt),
      endedAt: new Date(endedAt),
      route: null,
      petParticipants: [{ petId: petA }, { petId: petB }],
    });

    const result = await service.create(userId, {
      type: 'WALK',
      petIds: [petA, petB],
      startedAt,
      endedAt,
    });

    expect(result.id).toBe('activity-1');
    expect(households.assertPetAccessible).toHaveBeenCalledTimes(2);
    expect(households.resolveActivityHousehold).toHaveBeenCalledWith(
      userId,
      [petA, petB],
      undefined
    );
    expect(prisma.activity.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({
          userId,
          householdId,
          petId: petA,
          humanParticipants: {
            create: {
              userId,
              role: 'RECORDER',
            },
          },
          petParticipants: {
            create: [
              { petId: petA, metrics: undefined },
              { petId: petB, metrics: undefined },
            ],
          },
        }),
      })
    );
  });

  it('emits one trusted reward event per participating pet with stable dedupe keys', async () => {
    const { service, prisma, careEvents } = createHarness();
    prisma.activity.create.mockResolvedValue({
      id: 'activity-2',
      userId,
      householdId,
      petId: petA,
      type: 'WALK',
      startedAt: new Date('2026-08-21T17:00:00.000Z'),
      endedAt: new Date('2026-08-21T17:30:00.000Z'),
      route: null,
      petParticipants: [{ petId: petA }, { petId: petB }],
    });

    await service.create(userId, {
      type: 'WALK',
      petIds: [petA, petB],
      startedAt: '2026-08-21T17:00:00.000Z',
      endedAt: '2026-08-21T17:30:00.000Z',
    });

    expect(careEvents.record).toHaveBeenCalledTimes(2);
    expect(careEvents.record).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        userId,
        petId: petA,
        pathway: 'MOVE',
        dedupeKey: 'activity:activity-2:completed',
        context: expect.objectContaining({ participantCount: 2 }),
      })
    );
    expect(careEvents.record).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        userId,
        petId: petB,
        pathway: 'MOVE',
        dedupeKey: `activity:activity-2:pet:${petB}:completed`,
      })
    );
  });

  it('preserves legacy single-pet input while moving storage to participant rows', async () => {
    const { service, prisma } = createHarness();
    prisma.activity.create.mockResolvedValue({
      id: 'activity-legacy',
      userId,
      householdId,
      petId: petA,
      type: 'PLAY',
      startedAt: new Date('2026-08-21T17:00:00.000Z'),
      endedAt: null,
      route: null,
      petParticipants: [{ petId: petA }],
    });

    await service.create(userId, {
      type: 'PLAY',
      petId: petA,
      startedAt: '2026-08-21T17:00:00.000Z',
    });

    expect(prisma.activity.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({
          petId: petA,
          petParticipants: {
            create: [{ petId: petA, metrics: undefined }],
          },
        }),
      })
    );
  });

  it('rejects impossible activity chronology before writing an activity', async () => {
    const { service, prisma } = createHarness();

    await expect(
      service.create(userId, {
        type: 'WALK',
        petIds: [petA],
        startedAt: '2026-08-21T18:00:00.000Z',
        endedAt: '2026-08-21T17:00:00.000Z',
      })
    ).rejects.toBeInstanceOf(BadRequestException);

    expect(prisma.activity.create).not.toHaveBeenCalled();
  });
});
