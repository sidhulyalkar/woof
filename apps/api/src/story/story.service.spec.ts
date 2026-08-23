import { NotFoundException } from '@nestjs/common';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { StoryService } from './story.service';

const userId = '11111111-1111-4111-8111-111111111111';
const petId = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const activityId = '22222222-2222-4222-8222-222222222222';
const careId = '33333333-3333-4333-8333-333333333333';
const mediaId = '44444444-4444-4444-8444-444444444444';

function activity(overrides: Record<string, unknown> = {}) {
  return {
    id: activityId,
    petId,
    type: 'WALK',
    startedAt: new Date('2026-08-20T17:00:00.000Z'),
    endedAt: new Date('2026-08-20T17:45:00.000Z'),
    route: null,
    humanMetrics: null,
    petMetrics: null,
    jointMetrics: null,
    pet: { id: petId, name: 'Scout' },
    petParticipants: [{ petId, pet: { name: 'Scout' } }],
    ...overrides,
  };
}

function careEvent(overrides: Record<string, unknown> = {}) {
  return {
    id: careId,
    petId,
    eventType: 'QUEST_REFLECTION',
    pathway: 'BOND',
    occurredAt: new Date('2026-08-21T17:00:00.000Z'),
    source: 'ADVENTURE',
    context: {},
    outcome: { dogExperience: 'loved_it' },
    pet: { name: 'Scout' },
    ...overrides,
  };
}

function media(overrides: Record<string, unknown> = {}) {
  return {
    id: mediaId,
    petId,
    filename: 'beach-day.jpg',
    mediaType: 'image',
    capturedAt: new Date('2026-08-19T12:00:00.000Z'),
    createdAt: new Date('2026-08-19T12:05:00.000Z'),
    favorite: true,
    pet: { name: 'Scout' },
    ...overrides,
  };
}

function createHarness() {
  const txNotification = {
    create: jest.fn().mockResolvedValue({ id: 'curation-1' }),
    update: jest.fn().mockResolvedValue({ id: 'curation-1' }),
    delete: jest.fn().mockResolvedValue({ id: 'curation-1' }),
  };
  const tx = {
    $queryRaw: jest.fn().mockResolvedValue([]),
    notification: txNotification,
  };
  const prisma = {
    activity: {
      findMany: jest.fn().mockResolvedValue([]),
      count: jest.fn().mockResolvedValue(0),
      findFirst: jest.fn().mockResolvedValue({ id: activityId }),
    },
    careEvent: {
      findMany: jest.fn().mockResolvedValue([]),
      findFirst: jest.fn().mockResolvedValue({ id: careId }),
    },
    mediaAsset: {
      findMany: jest.fn().mockResolvedValue([]),
      count: jest.fn().mockResolvedValue(0),
      findFirst: jest.fn().mockResolvedValue(null),
    },
    notification: {
      findMany: jest.fn().mockResolvedValue([]),
    },
    householdPet: {
      findMany: jest.fn().mockResolvedValue([{ petId }]),
    },
    $transaction: jest.fn(async (callback: (client: typeof tx) => Promise<unknown>) =>
      callback(tx)
    ),
  };
  const households = {
    assertPetAccessible: jest.fn().mockResolvedValue({ id: petId }),
    householdActivityWhere: jest.fn().mockReturnValue({ userId }),
  };

  return {
    prisma,
    tx,
    txNotification,
    households,
    service: new StoryService(
      prisma as unknown as PrismaService,
      households as unknown as HouseholdsService
    ),
  };
}

describe('StoryService', () => {
  it('composes Activity, CareEvent, and Media sources into one chronological read model', async () => {
    const { service, prisma } = createHarness();
    prisma.activity.findMany.mockResolvedValue([activity()]);
    prisma.activity.count.mockResolvedValue(1);
    prisma.careEvent.findMany.mockResolvedValue([careEvent()]);
    prisma.mediaAsset.findMany.mockResolvedValueOnce([media()]).mockResolvedValueOnce([]);
    prisma.mediaAsset.count.mockResolvedValue(1);
    prisma.mediaAsset.findFirst
      .mockResolvedValueOnce({
        capturedAt: new Date('2026-08-19T12:00:00.000Z'),
        createdAt: new Date('2026-08-19T12:05:00.000Z'),
      })
      .mockResolvedValueOnce({
        capturedAt: new Date('2026-08-19T12:00:00.000Z'),
        createdAt: new Date('2026-08-19T12:05:00.000Z'),
      });

    const result = await service.getStory(userId, { petId });

    expect(result.moments.map((moment) => moment.sourceType)).toEqual([
      'CARE_EVENT',
      'ACTIVITY',
      'MEDIA',
    ]);
    expect(result.stats).toEqual(
      expect.objectContaining({ activities: 1, activeMinutes: 45, memories: 1 })
    );
  });

  it('merges captured and undated media by effective chronology before applying the source bound', async () => {
    const { service, prisma } = createHarness();
    const capturedId = '99999999-9999-4999-8999-999999999999';
    const undatedId = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee';
    prisma.mediaAsset.findMany
      .mockResolvedValueOnce([
        media({
          id: capturedId,
          capturedAt: new Date('2025-01-01T12:00:00.000Z'),
          createdAt: new Date('2025-01-02T12:00:00.000Z'),
        }),
      ])
      .mockResolvedValueOnce([
        media({
          id: undatedId,
          capturedAt: null,
          createdAt: new Date('2026-08-20T12:00:00.000Z'),
        }),
      ]);

    const result = await service.getStory(userId, {});

    expect(prisma.mediaAsset.findMany).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        where: expect.objectContaining({ capturedAt: { not: null } }),
        orderBy: { capturedAt: 'desc' },
      })
    );
    expect(prisma.mediaAsset.findMany).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        where: expect.objectContaining({ capturedAt: null }),
        orderBy: { createdAt: 'desc' },
      })
    );
    expect(
      result.moments
        .filter((moment) => moment.sourceType === 'MEDIA')
        .map((moment) => moment.sourceId)
    ).toEqual([undatedId, capturedId]);
  });

  it('does not duplicate an Activity-backed CareEvent or narrate tracker device upkeep', async () => {
    const { service, prisma } = createHarness();
    prisma.activity.findMany.mockResolvedValue([activity()]);
    prisma.activity.count.mockResolvedValue(1);
    prisma.careEvent.findMany.mockResolvedValue([
      careEvent({
        id: '55555555-5555-4555-8555-555555555555',
        eventType: 'ACTIVITY_WALK',
        context: { activityId },
      }),
      careEvent({
        id: '66666666-6666-4666-8666-666666666666',
        eventType: 'TRACKER_DEVICE_STATUS',
        source: 'AUTOPILOT_FI',
      }),
    ]);

    const result = await service.getStory(userId, {});

    expect(result.moments).toHaveLength(1);
    expect(result.moments[0]).toEqual(
      expect.objectContaining({ sourceType: 'ACTIVITY', sourceId: activityId })
    );
  });

  it('retains a wearable daily summary only as explicitly non-diagnostic context', async () => {
    const { service, prisma } = createHarness();
    prisma.careEvent.findMany.mockResolvedValue([
      careEvent({
        eventType: 'TRACKER_DAILY_ACTIVITY',
        pathway: 'MOVE',
        source: 'AUTOPILOT_FI',
        context: { activityMinutes: 62 },
        outcome: {},
      }),
    ]);

    const result = await service.getStory(userId, {});

    expect(result.moments[0]).toEqual(
      expect.objectContaining({
        sourceType: 'CARE_EVENT',
        title: 'Daily movement summary',
        summary: expect.stringContaining('not a health judgment'),
      })
    );
  });

  it('queries shared CareEvents only through HOUSEHOLD visibility while preserving own events', async () => {
    const { service, prisma } = createHarness();

    await service.getStory(userId, {});

    const where = prisma.careEvent.findMany.mock.calls[0]?.[0]?.where;
    expect(where).toEqual(
      expect.objectContaining({
        AND: expect.arrayContaining([
          expect.objectContaining({
            OR: expect.arrayContaining([
              { userId },
              expect.objectContaining({
                petId: { in: [petId] },
                visibility: 'HOUSEHOLD',
              }),
            ]),
          }),
        ]),
      })
    );
  });

  it('applies SAVED notes and removes HIDDEN sources without mutating source records', async () => {
    const { service, prisma } = createHarness();
    prisma.activity.findMany.mockResolvedValue([activity()]);
    prisma.activity.count.mockResolvedValue(1);
    prisma.mediaAsset.findMany.mockResolvedValueOnce([media()]).mockResolvedValueOnce([]);
    prisma.notification.findMany.mockResolvedValue([
      {
        id: 'curation-saved',
        payload: {
          schemaVersion: 'dogos-story-curation-v1',
          sourceType: 'ACTIVITY',
          sourceId: activityId,
          state: 'SAVED',
          note: 'The first sunset loop.',
          updatedAt: '2026-08-22T10:00:00.000Z',
        },
      },
      {
        id: 'curation-hidden',
        payload: {
          schemaVersion: 'dogos-story-curation-v1',
          sourceType: 'MEDIA',
          sourceId: mediaId,
          state: 'HIDDEN',
          updatedAt: '2026-08-22T10:01:00.000Z',
        },
      },
    ]);

    const result = await service.getStory(userId, {});

    expect(result.moments).toHaveLength(1);
    expect(result.moments[0].curation).toEqual({
      state: 'SAVED',
      note: 'The first sunset loop.',
    });
  });

  it('upserts one curation envelope under an advisory lock and clears it idempotently', async () => {
    const { service, tx, txNotification } = createHarness();

    const saved = await service.updateCuration(userId, {
      sourceType: 'ACTIVITY',
      sourceId: activityId,
      action: 'SAVE',
      note: 'Keep this one.',
    });

    expect(tx.$queryRaw).toHaveBeenCalled();
    expect(txNotification.create).toHaveBeenCalledWith({
      data: expect.objectContaining({
        userId,
        type: 'STORY_CURATION',
        payload: expect.objectContaining({
          sourceType: 'ACTIVITY',
          sourceId: activityId,
          state: 'SAVED',
          note: 'Keep this one.',
        }),
      }),
    });
    expect(saved.state).toBe('SAVED');

    tx.$queryRaw
      .mockResolvedValueOnce([{ acquired: 1 }])
      .mockResolvedValueOnce([{ id: 'curation-1', payload: {} }]);
    const cleared = await service.updateCuration(userId, {
      sourceType: 'ACTIVITY',
      sourceId: activityId,
      action: 'CLEAR',
    });
    expect(txNotification.delete).toHaveBeenCalledWith({ where: { id: 'curation-1' } });
    expect(cleared.state).toBeNull();
  });

  it('fails closed when the viewer cannot access the referenced source', async () => {
    const { service, prisma } = createHarness();
    prisma.mediaAsset.findFirst.mockResolvedValue(null);

    await expect(
      service.updateCuration(userId, {
        sourceType: 'MEDIA',
        sourceId: mediaId,
        action: 'SAVE',
      })
    ).rejects.toBeInstanceOf(NotFoundException);
  });

  it('counts only semantic place labels, never raw route coordinates, and reports bounded history honestly', async () => {
    const { service, prisma } = createHarness();
    prisma.activity.count.mockResolvedValue(5001);
    prisma.activity.findMany.mockResolvedValue([
      activity({
        id: '77777777-7777-4777-8777-777777777777',
        route: { coordinates: [[-122.1, 37.4]] },
        humanMetrics: { distanceMeters: 1200 },
      }),
      activity({
        id: '88888888-8888-4888-8888-888888888888',
        route: { placeName: 'Redwood Loop', coordinates: [[-122.2, 37.5]] },
        humanMetrics: { distanceKm: 2.5 },
      }),
    ]);

    const result = await service.getStory(userId, {});

    expect(result.stats.namedPlaces).toBe(1);
    expect(result.stats.distanceMeters).toBe(3700);
    expect(result.stats.coverage).toBe('BOUNDED');
  });

  it('timestamps the first-memory milestone from the real earliest READY media source', async () => {
    const { service, prisma } = createHarness();
    prisma.mediaAsset.count.mockResolvedValue(2);
    prisma.mediaAsset.findFirst
      .mockResolvedValueOnce({
        capturedAt: new Date('2025-04-02T10:00:00.000Z'),
        createdAt: new Date('2026-01-01T10:00:00.000Z'),
      })
      .mockResolvedValueOnce({
        capturedAt: null,
        createdAt: new Date('2025-06-01T10:00:00.000Z'),
      });

    const result = await service.getStory(userId, {});
    const firstMemory = result.milestones.find((milestone) => milestone.id === 'first-memory');

    expect(firstMemory?.achievedAt).toBe('2025-04-02T10:00:00.000Z');
  });
});
