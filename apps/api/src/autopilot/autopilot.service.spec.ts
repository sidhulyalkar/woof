import { CareEventsService } from '../care-events/care-events.service';
import { HouseholdsService } from '../households/households.service';
import { NotificationsService } from '../notifications/notifications.service';
import { PrismaService } from '../prisma/prisma.service';
import { AutopilotService } from './autopilot.service';

const userId = '11111111-1111-4111-8111-111111111111';
const petId = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';

function createHarness() {
  const txNotification = {
    findFirst: jest.fn(),
    create: jest.fn().mockImplementation(({ data }) =>
      Promise.resolve({
        id: 'signal-new',
        ...data,
        readAt: null,
        createdAt: new Date(),
      })
    ),
    update: jest.fn(),
  };
  const tx = {
    $queryRaw: jest.fn().mockResolvedValue([{ acquired: true }]),
    notification: txNotification,
  };
  const prisma = {
    pet: { findFirst: jest.fn().mockResolvedValue({ id: petId }) },
    notification: {
      create: jest.fn().mockImplementation(({ data }) =>
        Promise.resolve({
          id: `notification-${Math.random()}`,
          ...data,
          readAt: null,
          createdAt: new Date(),
        })
      ),
      findMany: jest.fn().mockResolvedValue([]),
      findFirst: jest.fn(),
      update: jest.fn(),
      updateMany: jest.fn(),
    },
    $queryRaw: jest.fn().mockResolvedValue([]),
    $transaction: jest.fn(async (callback: (client: typeof tx) => Promise<unknown>) =>
      callback(tx)
    ),
  };
  const careEvents = {
    record: jest.fn().mockResolvedValue({
      careEventId: 'care-event-1',
      ledgerId: 'ledger-1',
      bondXp: 0,
      pathway: 'MOVE',
      policyVersion: 'bond-xp-v1',
      explanation: 'not eligible',
      duplicate: false,
    }),
  };
  const households = {
    assertPetAccessible: jest.fn().mockResolvedValue({ id: petId }),
  };
  const notifications = {
    sendPushNotification: jest.fn().mockResolvedValue({ success: true }),
  };

  return {
    prisma,
    tx,
    txNotification,
    careEvents,
    households,
    notifications,
    service: new AutopilotService(
      prisma as unknown as PrismaService,
      careEvents as unknown as CareEventsService,
      households as unknown as HouseholdsService,
      notifications as unknown as NotificationsService
    ),
  };
}

describe('AutopilotService', () => {
  it('records wearable summaries as private zero-reward CareEvents', async () => {
    const { service, careEvents } = createHarness();

    const result = await service.ingestProviderObservation(userId, 'fi', {
      petId,
      externalEventId: 'fi-day-1',
      kind: 'DAILY_ACTIVITY',
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: { activityMinutes: 74, steps: 9200 },
    });

    expect(result.bondXp).toBe(0);
    expect(careEvents.record).toHaveBeenCalledWith(
      expect.objectContaining({
        userId,
        petId,
        eventType: 'TRACKER_DAILY_ACTIVITY',
        source: 'AUTOPILOT_FI',
        evidenceType: 'WEARABLE',
        visibility: 'PRIVATE',
        safetyEligible: false,
        dedupeKey: `autopilot:fi:${petId}:fi-day-1`,
        context: expect.objectContaining({
          privacyClass: 'SUMMARY_ONLY',
          nonDiagnostic: true,
          activityMinutes: 74,
        }),
      })
    );
  });

  it('repairs a missing signal when a provider replay finds the CareEvent already committed', async () => {
    const { service, careEvents, tx, txNotification } = createHarness();
    careEvents.record.mockResolvedValue({
      careEventId: 'care-event-existing',
      ledgerId: 'ledger-existing',
      bondXp: 0,
      pathway: 'CARE',
      policyVersion: 'bond-xp-v1',
      explanation: 'duplicate',
      duplicate: true,
    });
    txNotification.findFirst.mockResolvedValue(null);

    const result = await service.ingestProviderObservation(userId, 'tractive', {
      petId,
      externalEventId: 'same-event',
      kind: 'DEVICE_STATUS',
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: { batteryPercent: 9 },
    });

    expect(result.duplicate).toBe(true);
    expect(result.signal).toEqual(
      expect.objectContaining({
        id: 'signal-new',
        payload: expect.objectContaining({
          sourceCareEventId: 'care-event-existing',
          signalType: 'TRACKER_BATTERY_LOW',
        }),
      })
    );
    expect(tx.$queryRaw).toHaveBeenCalled();
    expect(txNotification.create).toHaveBeenCalledTimes(1);
  });

  it('returns an existing replay signal without duplicating it, even after acknowledgement', async () => {
    const { service, careEvents, txNotification } = createHarness();
    careEvents.record.mockResolvedValue({
      careEventId: 'care-event-existing',
      ledgerId: 'ledger-existing',
      bondXp: 0,
      pathway: 'CARE',
      policyVersion: 'bond-xp-v1',
      explanation: 'duplicate',
      duplicate: true,
    });
    txNotification.findFirst.mockResolvedValue({
      id: 'signal-existing',
      userId,
      type: 'AUTOPILOT_SIGNAL',
      readAt: new Date('2026-08-22T09:00:00.000Z'),
      createdAt: new Date('2026-08-22T08:00:01.000Z'),
      payload: {
        schemaVersion: 'dogos-autopilot-signal-v1',
        petId,
        sourceCareEventId: 'care-event-existing',
        signalType: 'TRACKER_BATTERY_LOW',
        level: 'INFO',
        title: 'Tracker battery is running low',
        body: 'Already acknowledged.',
        observedAt: '2026-08-22T08:00:00.000Z',
        evidence: { batteryPercent: 9, provider: 'TRACTIVE' },
        nonDiagnostic: true,
      },
    });

    const result = await service.ingestProviderObservation(userId, 'tractive', {
      petId,
      externalEventId: 'same-event',
      kind: 'DEVICE_STATUS',
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: { batteryPercent: 9 },
    });

    expect(result.signal).toEqual(
      expect.objectContaining({
        id: 'signal-existing',
        payload: expect.objectContaining({
          sourceCareEventId: 'care-event-existing',
          signalType: 'TRACKER_BATTERY_LOW',
        }),
      })
    );
    expect(txNotification.create).not.toHaveBeenCalled();
  });

  it('requires a meaningful baseline before producing a lower-activity check-in', async () => {
    const { service, prisma, txNotification } = createHarness();
    prisma.$queryRaw.mockResolvedValue([
      { activity_minutes: 80 },
      { activity_minutes: 75 },
      { activity_minutes: 78 },
      { activity_minutes: 82 },
      { activity_minutes: 76 },
    ]);

    const result = await service.ingestProviderObservation(userId, 'fi', {
      petId,
      externalEventId: 'fi-low-sparse',
      kind: 'DAILY_ACTIVITY',
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: { activityMinutes: 18 },
    });

    expect(result.signal).toBeNull();
    expect(txNotification.create).not.toHaveBeenCalled();
  });

  it('creates a non-diagnostic check-in only for a large drop against six or more prior summaries', async () => {
    const { service, prisma } = createHarness();
    prisma.$queryRaw.mockResolvedValue([
      { activity_minutes: 80 },
      { activity_minutes: 76 },
      { activity_minutes: 78 },
      { activity_minutes: 82 },
      { activity_minutes: 74 },
      { activity_minutes: 79 },
      { activity_minutes: 81 },
    ]);

    const result = await service.ingestProviderObservation(userId, 'tractive', {
      petId,
      externalEventId: 'tractive-low-1',
      kind: 'DAILY_ACTIVITY',
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: { activeMinutes: 25 },
    });

    expect(result.signal?.payload).toEqual(
      expect.objectContaining({
        signalType: 'ACTIVITY_BELOW_RECENT_BASELINE',
        level: 'CHECK_IN',
        nonDiagnostic: true,
        evidence: expect.objectContaining({
          currentActivityMinutes: 25,
          baselineSamples: 7,
          provider: 'TRACTIVE',
        }),
      })
    );
  });

  it('creates an informational low-battery signal without health claims', async () => {
    const { service } = createHarness();

    const result = await service.ingestProviderObservation(userId, 'fi', {
      petId,
      externalEventId: 'fi-battery-1',
      kind: 'DEVICE_STATUS',
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: { batteryPercent: 14, status: 'online' },
    });

    expect(result.signal?.payload).toEqual(
      expect.objectContaining({
        signalType: 'TRACKER_BATTERY_LOW',
        level: 'INFO',
        nonDiagnostic: true,
        evidence: { batteryPercent: 14, provider: 'FI' },
      })
    );
  });

  it('lets household participants schedule care reminders without changing pet state', async () => {
    const { service, prisma, households } = createHarness();

    const result = await service.createReminder(userId, {
      petId,
      kind: 'GROOMING',
      title: 'Brush coat',
      dueAt: '2026-08-24T18:00:00.000Z',
      repeatEveryDays: 7,
    });

    expect(households.assertPetAccessible).toHaveBeenCalledWith(userId, petId);
    expect(prisma.notification.create).toHaveBeenCalledWith({
      data: expect.objectContaining({
        userId,
        type: 'CARE_REMINDER',
        payload: expect.objectContaining({
          schemaVersion: 'dogos-care-reminder-v1',
          petId,
          kind: 'GROOMING',
          status: 'SCHEDULED',
          repeatEveryDays: 7,
        }),
      }),
    });
    expect(result.status).toBe('SCHEDULED');
  });

  it('does not mark a due reminder complete when push delivery fails', async () => {
    const { service, prisma, txNotification, notifications } = createHarness();
    const payload = {
      schemaVersion: 'dogos-care-reminder-v1',
      kind: 'MEDICATION',
      title: 'Medication reminder',
      petId,
      dueAt: '2026-08-21T18:00:00.000Z',
      status: 'SCHEDULED',
    } as const;
    const row = {
      id: 'reminder-1',
      userId,
      type: 'CARE_REMINDER',
      payload,
      readAt: null,
      createdAt: new Date('2026-08-21T17:00:00.000Z'),
    };
    prisma.notification.findMany.mockResolvedValue([row]);
    txNotification.findFirst.mockResolvedValue(row);
    notifications.sendPushNotification.mockResolvedValue({
      success: false,
      reason: 'no_subscription',
    });

    const result = await service.dispatchDueReminders();

    expect(result).toEqual({ attempted: 1, delivered: 0 });
    expect(txNotification.update).toHaveBeenCalledWith(
      expect.objectContaining({
        where: { id: 'reminder-1' },
        data: { payload: expect.objectContaining({ lastAttemptAt: expect.any(String) }) },
      })
    );
    expect(prisma.notification.update).not.toHaveBeenCalled();
  });
});
