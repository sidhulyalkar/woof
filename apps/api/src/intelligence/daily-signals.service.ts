import { BadRequestException, ConflictException, Injectable } from '@nestjs/common';
import { createHash } from 'node:crypto';
import type { CanonicalCareEventRecord } from '../care-events/care-event.types';
import { CareEventsService } from '../care-events/care-events.service';
import { HouseholdsService } from '../households/households.service';
import type { SignalDimension } from './baseline-policy-v1.types';
import { normalizeOwnerCheckinObservation } from './evidence-normalization-v1';
import { IntelligenceProjectionService } from './intelligence-projection.service';
import {
  dailySignalsDedupeKey,
  resolveDailySignalsTime,
} from './daily-signals-local-day-policy-v1';
import {
  DAILY_SIGNAL_CHOICES,
  DAILY_SIGNAL_DIMENSION_BY_FIELD,
  DAILY_SIGNAL_FIELDS,
  type DailySignalChoice,
  type DailySignalsAnswers,
  type DailySignalsCaptureReceipt,
} from './daily-signals.types';
import type { CreateDailySignalsDto } from './dto/daily-signals.dto';

export const DAILY_SIGNALS_CAPTURE_POLICY_V1 = Object.freeze({
  version: 'daily-signals-capture-v1' as const,
  eventType: 'DAILY_SIGNALS_CHECKIN' as const,
  source: 'INTELLIGENCE' as const,
  evidenceConfidence: 0.8,
  visibility: 'PRIVATE' as const,
  dedupeScope: 'PET' as const,
  noteMaxLength: 500,
});

type CanonicalDailySignalsPayload = {
  signals: DailySignalsAnswers;
  note?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isDailySignalChoice(value: unknown): value is DailySignalChoice {
  return DAILY_SIGNAL_CHOICES.includes(value as DailySignalChoice);
}

function normalizedAnswers(input: CreateDailySignalsDto['signals']): DailySignalsAnswers {
  if (!input || !isRecord(input)) {
    throw new BadRequestException('Daily Signals requires a signals object');
  }

  const answers: DailySignalsAnswers = {};
  for (const field of DAILY_SIGNAL_FIELDS) {
    const value = input[field];
    if (value === undefined) continue;
    if (!isDailySignalChoice(value)) {
      throw new BadRequestException(`Daily Signals contains an invalid ${field} answer`);
    }
    answers[field] = value;
  }
  if (Object.keys(answers).length === 0) {
    throw new BadRequestException('Daily Signals requires at least one answered dimension');
  }
  return answers;
}

function normalizedNote(note?: string): string | undefined {
  if (note === undefined) return undefined;
  if (typeof note !== 'string') throw new BadRequestException('Daily Signals note must be text');
  const trimmed = note.trim();
  if (trimmed.length > DAILY_SIGNALS_CAPTURE_POLICY_V1.noteMaxLength) {
    throw new BadRequestException(
      `Daily Signals note must be ${DAILY_SIGNALS_CAPTURE_POLICY_V1.noteMaxLength} characters or fewer`
    );
  }
  return trimmed || undefined;
}

function canonicalPayloadHash(payload: CanonicalDailySignalsPayload): string {
  const orderedSignals: DailySignalsAnswers = {};
  for (const field of DAILY_SIGNAL_FIELDS) {
    const value = payload.signals[field];
    if (value !== undefined) orderedSignals[field] = value;
  }
  return createHash('sha256')
    .update(JSON.stringify({ signals: orderedSignals, note: payload.note ?? null }))
    .digest('hex');
}

function parseCanonicalAnswers(event: CanonicalCareEventRecord): DailySignalsAnswers {
  const rawSignals = event.outcome.signals;
  if (!isRecord(rawSignals)) {
    throw new ConflictException('Canonical Daily Signals event is missing structured answers');
  }

  const answers: DailySignalsAnswers = {};
  for (const field of DAILY_SIGNAL_FIELDS) {
    const value = rawSignals[field];
    if (value === undefined) continue;
    if (!isDailySignalChoice(value)) {
      throw new ConflictException('Canonical Daily Signals event contains an invalid answer');
    }
    answers[field] = value;
  }
  if (Object.keys(answers).length === 0) {
    throw new ConflictException('Canonical Daily Signals event has no usable answers');
  }
  return answers;
}

@Injectable()
export class DailySignalsService {
  constructor(
    private readonly households: HouseholdsService,
    private readonly careEvents: CareEventsService,
    private readonly projection: IntelligenceProjectionService
  ) {}

  async capture(
    userId: string,
    dto: CreateDailySignalsDto,
    now: Date = new Date()
  ): Promise<DailySignalsCaptureReceipt> {
    const household = await this.households.assertHouseholdPetAccessible(
      userId,
      dto.householdId,
      dto.petId
    );
    if (!household.timezone) {
      throw new BadRequestException(
        'Household timezone is required before Daily Signals can be recorded'
      );
    }

    let resolvedTime;
    try {
      resolvedTime = resolveDailySignalsTime({
        requestedObservedAt: dto.observedAt,
        householdTimezone: household.timezone,
        now,
      });
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : 'Invalid Daily Signals time'
      );
    }

    const signals = normalizedAnswers(dto.signals);
    const note = normalizedNote(dto.note);
    const payloadHash = canonicalPayloadHash({ signals, ...(note ? { note } : {}) });
    const dedupeKey = dailySignalsDedupeKey({
      householdId: household.householdId,
      petId: dto.petId,
      localDate: resolvedTime.localDate,
    });

    const rewardReceipt = await this.careEvents.record({
      userId,
      petId: dto.petId,
      eventType: DAILY_SIGNALS_CAPTURE_POLICY_V1.eventType,
      pathway: 'CARE',
      occurredAt: new Date(resolvedTime.observedAt),
      source: DAILY_SIGNALS_CAPTURE_POLICY_V1.source,
      evidenceType: 'SELF_REPORT',
      evidenceConfidence: DAILY_SIGNALS_CAPTURE_POLICY_V1.evidenceConfidence,
      dedupeKey,
      dedupeScope: DAILY_SIGNALS_CAPTURE_POLICY_V1.dedupeScope,
      visibility: DAILY_SIGNALS_CAPTURE_POLICY_V1.visibility,
      safetyEligible: false,
      context: {
        householdId: household.householdId,
        localDate: resolvedTime.localDate,
        timezone: resolvedTime.timezone,
        capturePolicyVersion: DAILY_SIGNALS_CAPTURE_POLICY_V1.version,
        payloadHash,
      },
      outcome: {
        signals,
        ...(note ? { note } : {}),
      },
    });

    const canonical = await this.careEvents.getAuthorizedEvent(userId, rewardReceipt.careEventId);
    const canonicalHash = canonical.context.payloadHash;
    if (canonicalHash !== payloadHash) {
      throw new ConflictException(
        'Daily Signals is already recorded for this pet and local day with different answers'
      );
    }
    if (
      canonical.eventType !== DAILY_SIGNALS_CAPTURE_POLICY_V1.eventType ||
      canonical.source !== DAILY_SIGNALS_CAPTURE_POLICY_V1.source ||
      canonical.evidenceType !== 'SELF_REPORT' ||
      canonical.petId !== dto.petId ||
      canonical.context.householdId !== household.householdId ||
      canonical.context.localDate !== resolvedTime.localDate
    ) {
      throw new ConflictException(
        'Canonical Daily Signals identity does not match the requested capture'
      );
    }

    const projectedDimensions = await this.replayCanonicalDailySignalsEvent(userId, canonical);
    const canonicalTimezone = canonical.context.timezone;
    if (typeof canonicalTimezone !== 'string') {
      throw new ConflictException(
        'Canonical Daily Signals event is missing its household timezone'
      );
    }

    return {
      careEventId: canonical.id,
      householdId: household.householdId,
      petId: dto.petId,
      localDate: resolvedTime.localDate,
      timezone: canonicalTimezone,
      duplicate: rewardReceipt.duplicate,
      futureTimestampNormalized: resolvedTime.futureTimestampNormalized,
      projectedDimensions,
    };
  }

  async replay(userId: string, careEventId: string): Promise<SignalDimension[]> {
    const canonical = await this.careEvents.getAuthorizedEvent(userId, careEventId);
    return this.replayCanonicalDailySignalsEvent(userId, canonical);
  }

  private async replayCanonicalDailySignalsEvent(
    requestingUserId: string,
    event: CanonicalCareEventRecord
  ): Promise<SignalDimension[]> {
    if (
      event.eventType !== DAILY_SIGNALS_CAPTURE_POLICY_V1.eventType ||
      event.source !== DAILY_SIGNALS_CAPTURE_POLICY_V1.source ||
      event.evidenceType !== 'SELF_REPORT' ||
      !event.petId
    ) {
      throw new ConflictException('CareEvent is not a canonical Daily Signals source');
    }

    const localDate = event.context.localDate;
    const householdId = event.context.householdId;
    if (typeof localDate !== 'string' || typeof householdId !== 'string') {
      throw new ConflictException('Canonical Daily Signals event is missing capture context');
    }
    await this.households.assertHouseholdPetAccessible(requestingUserId, householdId, event.petId);

    const answers = parseCanonicalAnswers(event);
    const projectedDimensions: SignalDimension[] = [];
    for (const field of DAILY_SIGNAL_FIELDS) {
      const choice = answers[field];
      if (choice === undefined || choice === 'UNSURE') continue;
      const dimension = DAILY_SIGNAL_DIMENSION_BY_FIELD[field];
      const candidate = normalizeOwnerCheckinObservation({
        userId: event.userId,
        petId: event.petId,
        careEventId: event.id,
        dimension,
        choice,
        observedAt: event.occurredAt,
        localDate,
        confidence: event.evidenceConfidence,
      });
      if (!candidate) continue;
      await this.projection.projectObservation(candidate);
      projectedDimensions.push(dimension);
    }
    return projectedDimensions;
  }
}
