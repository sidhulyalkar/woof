import { Injectable, NotFoundException } from '@nestjs/common';
import { Cron } from '@nestjs/schedule';
import { Prisma } from '@woof/database';
import { CareEventsService } from '../care-events/care-events.service';
import { HouseholdsService } from '../households/households.service';
import { NotificationsService } from '../notifications/notifications.service';
import { PrismaService } from '../prisma/prisma.service';
import type { CreateCareReminderDto, IngestTrackerObservationDto } from './dto/autopilot.dto';
import { normalizeProviderObservation } from './provider-adapters';
import type {
  AutopilotSignalPayload,
  CareReminderPayload,
  NormalizedTrackerObservation,
} from './autopilot.types';

type ActivityBaselineRow = {
  activity_minutes: number | null;
};

type NotificationRow = {
  id: string;
  type: string;
  payload: Prisma.JsonValue;
  readAt: Date | null;
  createdAt: Date;
};

function jsonObject(value: Prisma.JsonValue): Prisma.JsonObject | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return value as Prisma.JsonObject;
}

function readString(value: Prisma.JsonValue | undefined): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function readNumber(value: Prisma.JsonValue | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function readReminderPayload(value: Prisma.JsonValue): CareReminderPayload | null {
  const object = jsonObject(value);
  if (!object || object.schemaVersion !== 'dogos-care-reminder-v1') return null;

  const kind = readString(object.kind);
  const title = readString(object.title);
  const dueAt = readString(object.dueAt);
  const status = readString(object.status);
  if (
    !kind ||
    !['VET_APPOINTMENT', 'MEDICATION', 'GROOMING', 'GENERAL_CARE'].includes(kind) ||
    !title ||
    !dueAt ||
    !status ||
    !['SCHEDULED', 'COMPLETED', 'CANCELLED'].includes(status)
  ) {
    return null;
  }

  const repeatEveryDays = readNumber(object.repeatEveryDays);
  return {
    schemaVersion: 'dogos-care-reminder-v1',
    kind: kind as CareReminderPayload['kind'],
    title,
    ...(readString(object.detail) ? { detail: readString(object.detail) } : {}),
    ...(readString(object.petId) ? { petId: readString(object.petId) } : {}),
    dueAt,
    ...(repeatEveryDays !== undefined ? { repeatEveryDays } : {}),
    status: status as CareReminderPayload['status'],
    ...(readString(object.lastAttemptAt)
      ? { lastAttemptAt: readString(object.lastAttemptAt) }
      : {}),
    ...(readString(object.lastDeliveredAt)
      ? { lastDeliveredAt: readString(object.lastDeliveredAt) }
      : {}),
  };
}

function readSignalPayload(value: Prisma.JsonValue): AutopilotSignalPayload | null {
  const object = jsonObject(value);
  if (!object || object.schemaVersion !== 'dogos-autopilot-signal-v1') return null;

  const petId = readString(object.petId);
  const sourceCareEventId = readString(object.sourceCareEventId);
  const signalType = readString(object.signalType);
  const level = readString(object.level);
  const title = readString(object.title);
  const body = readString(object.body);
  const observedAt = readString(object.observedAt);
  const evidence = object.evidence;
  if (
    !petId ||
    !sourceCareEventId ||
    !signalType ||
    !['ACTIVITY_BELOW_RECENT_BASELINE', 'TRACKER_BATTERY_LOW'].includes(signalType) ||
    !level ||
    !['CHECK_IN', 'INFO'].includes(level) ||
    !title ||
    !body ||
    !observedAt ||
    !evidence ||
    typeof evidence !== 'object' ||
    Array.isArray(evidence)
  ) {
    return null;
  }

  const safeEvidence: Record<string, string | number | boolean> = {};
  for (const [key, valueEntry] of Object.entries(evidence)) {
    if (
      typeof valueEntry === 'string' ||
      typeof valueEntry === 'number' ||
      typeof valueEntry === 'boolean'
    ) {
      safeEvidence[key] = valueEntry;
    }
  }

  return {
    schemaVersion: 'dogos-autopilot-signal-v1',
    petId,
    sourceCareEventId,
    signalType: signalType as AutopilotSignalPayload['signalType'],
    level: level as AutopilotSignalPayload['level'],
    title,
    body,
    observedAt,
    evidence: safeEvidence,
    nonDiagnostic: true,
  };
}

function inputJson(value: CareReminderPayload | AutopilotSignalPayload): Prisma.InputJsonObject {
  return value as unknown as Prisma.InputJsonObject;
}

@Injectable()
export class AutopilotService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly careEvents: CareEventsService,
    private readonly households: HouseholdsService,
    private readonly notifications: NotificationsService,
  ) {}

  async getDashboard(userId: string) {
    const rows = await this.prisma.notification.findMany({
      where: {
        userId,
        type: { in: ['CARE_REMINDER', 'AUTOPILOT_SIGNAL'] },
        readAt: null,
      },
      orderBy: { createdAt: 'desc' },
      take: 100,
    });

    const reminders = rows
      .filter((row) => row.type === 'CARE_REMINDER')
      .map((row) => ({ id: row.id, ...readReminderPayload(row.payload) }))
      .filter((row): row is { id: string } & CareReminderPayload => row.schemaVersion !== undefined)
      .sort((a, b) => new Date(a.dueAt).getTime() - new Date(b.dueAt).getTime());

    const signals = rows
      .filter((row) => row.type === 'AUTOPILOT_SIGNAL')
      .map((row) => ({ id: row.id, ...readSignalPayload(row.payload) }))
      .filter((row): row is { id: string } & AutopilotSignalPayload => row.schemaVersion !== undefined);

    return {
      providers: [
        { provider: 'FI', status: 'STUB_READY', accepts: ['DAILY_ACTIVITY', 'DEVICE_STATUS'] },
        { provider: 'TRACTIVE', status: 'STUB_READY', accepts: ['DAILY_ACTIVITY', 'DEVICE_STATUS'] },
      ],
      reminders,
      signals,
      boundaries: {
        locationTelemetryStored: false,
        canonicalPetMutationAllowed: false,
        trackerObservationsRewardEligible: false,
        signalsDiagnostic: false,
      },
    };
  }

  async ingestProviderObservation(
    userId: string,
    provider: string,
    dto: IngestTrackerObservationDto,
  ) {
    // Provider connections belong to the pet owner in Phase A. Shared-household
    // scheduling is allowed, but an invited member cannot impersonate a tracker owner.
    const ownedPet = await this.prisma.pet.findFirst({
      where: { id: dto.petId, ownerId: userId },
      select: { id: true },
    });
    if (!ownedPet) throw new NotFoundException('Pet not found');

    const observation = normalizeProviderObservation(provider, dto);
    const eventType =
      observation.kind === 'DAILY_ACTIVITY' ? 'TRACKER_DAILY_ACTIVITY' : 'TRACKER_DEVICE_STATUS';
    const pathway = observation.kind === 'DAILY_ACTIVITY' ? 'MOVE' : 'CARE';
    const dedupeKey = `autopilot:${observation.provider.toLowerCase()}:${dto.petId}:${observation.externalEventId}`;

    const receipt = await this.careEvents.record({
      userId,
      petId: dto.petId,
      eventType,
      pathway,
      occurredAt: observation.observedAt,
      source: `AUTOPILOT_${observation.provider}`,
      evidenceType: 'WEARABLE',
      evidenceConfidence: 0.82,
      dedupeKey,
      visibility: 'PRIVATE',
      safetyEligible: false,
      context: {
        provider: observation.provider,
        observationKind: observation.kind,
        privacyClass: 'SUMMARY_ONLY',
        nonDiagnostic: true,
        ...observation.metrics,
      },
    });

    // Signal derivation is deliberately replay-safe. If the immutable CareEvent
    // was committed but the request died before signal persistence, a provider
    // retry repairs the missing signal. Existing signals are returned rather
    // than duplicated, including signals the user has already acknowledged.
    const signal = await this.maybeCreateSignal(
      userId,
      dto.petId,
      receipt.careEventId,
      observation,
    );

    return {
      careEventId: receipt.careEventId,
      duplicate: receipt.duplicate,
      bondXp: receipt.bondXp,
      observation,
      signal,
    };
  }

  async createReminder(userId: string, dto: CreateCareReminderDto) {
    if (dto.petId) await this.households.assertPetAccessible(userId, dto.petId);

    const payload: CareReminderPayload = {
      schemaVersion: 'dogos-care-reminder-v1',
      kind: dto.kind,
      title: dto.title.trim(),
      ...(dto.detail?.trim() ? { detail: dto.detail.trim() } : {}),
      ...(dto.petId ? { petId: dto.petId } : {}),
      dueAt: new Date(dto.dueAt).toISOString(),
      ...(dto.repeatEveryDays ? { repeatEveryDays: dto.repeatEveryDays } : {}),
      status: 'SCHEDULED',
    };

    const reminder = await this.prisma.notification.create({
      data: {
        userId,
        type: 'CARE_REMINDER',
        payload: inputJson(payload),
      },
    });

    return { id: reminder.id, ...payload };
  }

  async cancelReminder(userId: string, reminderId: string) {
    const reminder = await this.prisma.notification.findFirst({
      where: { id: reminderId, userId, type: 'CARE_REMINDER', readAt: null },
    });
    if (!reminder) throw new NotFoundException('Reminder not found');

    const payload = readReminderPayload(reminder.payload);
    if (!payload) throw new NotFoundException('Reminder not found');
    const cancelled: CareReminderPayload = { ...payload, status: 'CANCELLED' };
    await this.prisma.notification.update({
      where: { id: reminder.id },
      data: { readAt: new Date(), payload: inputJson(cancelled) },
    });
    return { success: true };
  }

  async acknowledgeSignal(userId: string, signalId: string) {
    const updated = await this.prisma.notification.updateMany({
      where: { id: signalId, userId, type: 'AUTOPILOT_SIGNAL', readAt: null },
      data: { readAt: new Date() },
    });
    if (updated.count === 0) throw new NotFoundException('Signal not found');
    return { success: true };
  }

  @Cron('0 */10 * * * *')
  async dispatchDueReminders() {
    const now = new Date();
    const rows = await this.prisma.notification.findMany({
      where: { type: 'CARE_REMINDER', readAt: null },
      orderBy: { createdAt: 'asc' },
      take: 200,
    });

    let attempted = 0;
    let delivered = 0;
    for (const row of rows) {
      const claimed = await this.claimDueReminder(row, now);
      if (!claimed) continue;
      attempted += 1;

      const delivery = await this.notifications.sendPushNotification({
        userId: claimed.userId,
        title: claimed.payload.title,
        body: claimed.payload.detail ?? this.reminderBody(claimed.payload.kind),
        url: '/notifications',
        data: {
          type: 'care_reminder',
          reminderId: claimed.id,
          petId: claimed.payload.petId,
          kind: claimed.payload.kind,
        },
      });

      if (!delivery.success) continue;
      delivered += 1;
      await this.completeReminderDelivery(claimed.id, claimed.payload, now);
    }

    return { attempted, delivered };
  }

  private async maybeCreateSignal(
    userId: string,
    petId: string,
    careEventId: string,
    observation: NormalizedTrackerObservation,
  ) {
    if (observation.kind === 'DEVICE_STATUS') {
      const battery = observation.metrics.batteryPercent;
      if (battery !== undefined && battery <= 15) {
        return this.createSignal(userId, {
          schemaVersion: 'dogos-autopilot-signal-v1',
          petId,
          sourceCareEventId: careEventId,
          signalType: 'TRACKER_BATTERY_LOW',
          level: 'INFO',
          title: 'Tracker battery is running low',
          body: 'The connected tracker reported a low battery. Charging it can keep future summaries complete.',
          observedAt: observation.observedAt.toISOString(),
          evidence: { batteryPercent: battery, provider: observation.provider },
          nonDiagnostic: true,
        });
      }
      return null;
    }

    const current = observation.metrics.activityMinutes;
    if (current === undefined) return null;
    const windowStart = new Date(observation.observedAt.getTime() - 28 * 24 * 60 * 60 * 1000);
    const baselineRows = await this.prisma.$queryRaw<ActivityBaselineRow[]>(Prisma.sql`
      SELECT (context->>'activityMinutes')::double precision AS activity_minutes
      FROM care_events
      WHERE user_id = ${userId}
        AND pet_id = ${petId}
        AND event_type = 'TRACKER_DAILY_ACTIVITY'
        AND id <> ${careEventId}
        AND occurred_at >= ${windowStart}
        AND occurred_at < ${observation.observedAt}
        AND context ? 'activityMinutes'
      ORDER BY occurred_at DESC
      LIMIT 14
    `);

    const samples = baselineRows
      .map((row) => row.activity_minutes)
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
    if (samples.length < 6) return null;

    const baseline = this.median(samples);
    const drop = baseline - current;
    if (!(current <= baseline * 0.55 && drop >= 20)) return null;

    return this.createSignal(userId, {
      schemaVersion: 'dogos-autopilot-signal-v1',
      petId,
      sourceCareEventId: careEventId,
      signalType: 'ACTIVITY_BELOW_RECENT_BASELINE',
      level: 'CHECK_IN',
      title: 'Activity looks lower than the recent tracker baseline',
      body: 'This can happen with rest days, weather, routine changes, or tracker wear. Check the context if the change seems meaningful. This is not a diagnosis.',
      observedAt: observation.observedAt.toISOString(),
      evidence: {
        currentActivityMinutes: Math.round(current),
        baselineActivityMinutes: Math.round(baseline),
        baselineSamples: samples.length,
        provider: observation.provider,
      },
      nonDiagnostic: true,
    });
  }

  private async createSignal(userId: string, payload: AutopilotSignalPayload) {
    return this.prisma.$transaction(async (tx) => {
      const lockKey = `dogos-autopilot-signal:${payload.sourceCareEventId}`;
      await tx.$queryRaw<Array<{ acquired: number }>>(Prisma.sql`
        WITH lock_row AS MATERIALIZED (
          SELECT pg_advisory_xact_lock(hashtextextended(${lockKey}, 0))
        )
        SELECT 1::int AS acquired FROM lock_row
      `);

      const existing = await tx.notification.findFirst({
        where: {
          userId,
          type: 'AUTOPILOT_SIGNAL',
          payload: {
            path: ['sourceCareEventId'],
            equals: payload.sourceCareEventId,
          },
        },
      });
      if (existing) {
        return {
          id: existing.id,
          payload: readSignalPayload(existing.payload) ?? payload,
        };
      }

      const notification = await tx.notification.create({
        data: {
          userId,
          type: 'AUTOPILOT_SIGNAL',
          payload: inputJson(payload),
        },
      });
      return { id: notification.id, payload };
    });
  }

  private async claimDueReminder(row: NotificationRow, now: Date) {
    return this.prisma.$transaction(async (tx) => {
      const lockKey = `dogos-autopilot-reminder:${row.id}`;
      const lock = await tx.$queryRaw<Array<{ acquired: boolean }>>(Prisma.sql`
        SELECT pg_try_advisory_xact_lock(hashtextextended(${lockKey}, 0)) AS acquired
      `);
      if (!lock[0]?.acquired) return null;

      const current = await tx.notification.findFirst({
        where: { id: row.id, type: 'CARE_REMINDER', readAt: null },
      });
      if (!current) return null;
      const payload = readReminderPayload(current.payload);
      if (!payload || payload.status !== 'SCHEDULED') return null;

      const dueAt = new Date(payload.dueAt);
      if (!Number.isFinite(dueAt.getTime()) || dueAt > now) return null;
      if (payload.lastAttemptAt) {
        const retryAfter = new Date(payload.lastAttemptAt).getTime() + 6 * 60 * 60 * 1000;
        if (retryAfter > now.getTime()) return null;
      }

      const claimedPayload: CareReminderPayload = {
        ...payload,
        lastAttemptAt: now.toISOString(),
      };
      await tx.notification.update({
        where: { id: current.id },
        data: { payload: inputJson(claimedPayload) },
      });

      return { id: current.id, userId: current.userId, payload: claimedPayload };
    });
  }

  private async completeReminderDelivery(id: string, payload: CareReminderPayload, deliveredAt: Date) {
    if (payload.repeatEveryDays) {
      const stepMs = payload.repeatEveryDays * 24 * 60 * 60 * 1000;
      let nextDueAt = new Date(payload.dueAt).getTime() + stepMs;
      while (nextDueAt <= deliveredAt.getTime()) nextDueAt += stepMs;
      const next: CareReminderPayload = {
        ...payload,
        dueAt: new Date(nextDueAt).toISOString(),
        lastDeliveredAt: deliveredAt.toISOString(),
        status: 'SCHEDULED',
      };
      await this.prisma.notification.update({
        where: { id },
        data: { payload: inputJson(next) },
      });
      return;
    }

    const completed: CareReminderPayload = {
      ...payload,
      status: 'COMPLETED',
      lastDeliveredAt: deliveredAt.toISOString(),
    };
    await this.prisma.notification.update({
      where: { id },
      data: { payload: inputJson(completed), readAt: deliveredAt },
    });
  }

  private reminderBody(kind: CareReminderPayload['kind']) {
    switch (kind) {
      case 'VET_APPOINTMENT':
        return 'A vet-related reminder is due.';
      case 'MEDICATION':
        return 'A medication reminder is due. Follow the instructions provided by your veterinarian.';
      case 'GROOMING':
        return 'A grooming reminder is due.';
      default:
        return 'A dogOS care reminder is due.';
    }
  }

  private median(values: number[]) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[middle - 1] + sorted[middle]) / 2
      : sorted[middle];
  }
}
