import {
  BadRequestException,
  ConflictException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { Prisma } from '@woof/database';
import { createHash } from 'node:crypto';
import type { WellbeingPathway } from '../care-events/care-event.types';
import { CareEventsService } from '../care-events/care-events.service';
import { localDateInTimeZone } from '../common/time/iana-timezone';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import type { SignalDimension } from './baseline-policy-v1.types';
import { DailySignalsService } from './daily-signals.service';
import {
  DAILY_SIGNAL_CHOICES,
  DAILY_SIGNAL_DIMENSION_BY_FIELD,
  DAILY_SIGNAL_FIELDS,
  type DailySignalChoice,
  type DailySignalsAnswers,
} from './daily-signals.types';
import type {
  CorrectDailySignalsDto,
  DailySignalsCurrentQueryDto,
} from './dto/daily-signals-correction.dto';
import { normalizeOwnerCheckinObservation } from './evidence-normalization-v1';
import { IntelligenceProjectionService } from './intelligence-projection.service';

export const DAILY_SIGNALS_CORRECTION_POLICY_V1 = Object.freeze({
  version: 'daily-signals-correction-v1' as const,
  eventType: 'DAILY_SIGNALS_CORRECTION' as const,
  source: 'INTELLIGENCE' as const,
  evidenceConfidence: 0.8,
  visibility: 'PRIVATE' as const,
  dedupeScope: 'PET' as const,
});

type DailySignalsEventRow = {
  id: string;
  user_id: string;
  pet_id: string | null;
  event_type: string;
  pathway: WellbeingPathway;
  occurred_at: Date;
  source: string;
  evidence_type: string | null;
  evidence_confidence: number;
  context: Record<string, unknown> | null;
  outcome: Record<string, unknown> | null;
  dedupe_key: string;
  visibility: 'PRIVATE' | 'HOUSEHOLD' | 'FRIENDS';
  created_at: Date;
};

type ProjectionRow = {
  id: string;
  source_event_id: string | null;
  dimension: SignalDimension;
  retracted_at: Date | null;
};

type ChainEvent = {
  id: string;
  userId: string;
  petId: string;
  eventType: 'DAILY_SIGNALS_CHECKIN' | 'DAILY_SIGNALS_CORRECTION';
  occurredAt: string;
  evidenceConfidence: number;
  sequence: number;
  correctsCareEventId: string | null;
  payloadHash: string | null;
  signals: DailySignalsAnswers;
};

type DailySignalsChain = {
  householdId: string;
  petId: string;
  localDate: string;
  timezone: string;
  root: ChainEvent;
  events: ChainEvent[];
};

export type DailySignalsEffectiveState = {
  rootCareEventId: string;
  currentCareEventId: string;
  correctionSequence: number;
  householdId: string;
  petId: string;
  localDate: string;
  timezone: string;
  status: 'ORIGINAL' | 'CORRECTED';
  signals: DailySignalsAnswers;
};

export type DailySignalsCorrectionReceipt = {
  correctionCareEventId: string;
  duplicate: boolean;
  currentAdvanced: boolean;
  state: DailySignalsEffectiveState;
  projectedDimensions: SignalDimension[];
  retractedDimensions: SignalDimension[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isDailySignalChoice(value: unknown): value is DailySignalChoice {
  return DAILY_SIGNAL_CHOICES.includes(value as DailySignalChoice);
}

function normalizeSignals(value: unknown, requireOne: boolean): DailySignalsAnswers {
  if (!isRecord(value)) {
    throw new BadRequestException('Daily Signals correction requires a signals object');
  }

  const signals: DailySignalsAnswers = {};
  for (const field of DAILY_SIGNAL_FIELDS) {
    const choice = value[field];
    if (choice === undefined) continue;
    if (!isDailySignalChoice(choice)) {
      throw new BadRequestException(`Daily Signals contains an invalid ${field} answer`);
    }
    signals[field] = choice;
  }

  if (requireOne && Object.keys(signals).length === 0) {
    throw new ConflictException('Canonical Daily Signals event has no usable answers');
  }
  return signals;
}

function structuredPayloadHash(signals: DailySignalsAnswers) {
  const ordered: DailySignalsAnswers = {};
  for (const field of DAILY_SIGNAL_FIELDS) {
    const choice = signals[field];
    if (choice !== undefined) ordered[field] = choice;
  }
  return createHash('sha256')
    .update(JSON.stringify({ signals: ordered }))
    .digest('hex');
}

function stringContext(context: Record<string, unknown>, key: string): string | null {
  const value = context[key];
  return typeof value === 'string' && value ? value : null;
}

function integerContext(context: Record<string, unknown>, key: string): number | null {
  const value = context[key];
  return Number.isInteger(value) ? (value as number) : null;
}

@Injectable()
export class DailySignalsCorrectionService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly households: HouseholdsService,
    private readonly careEvents: CareEventsService,
    private readonly dailySignals: DailySignalsService,
    private readonly projection: IntelligenceProjectionService
  ) {}

  async getCurrent(
    userId: string,
    query: DailySignalsCurrentQueryDto,
    now: Date = new Date()
  ): Promise<DailySignalsEffectiveState | null> {
    const identity = await this.resolveIdentity(userId, query, now);
    const chain = await this.loadChain(identity);
    return chain ? this.toEffectiveState(chain) : null;
  }

  async correct(
    userId: string,
    dto: CorrectDailySignalsDto,
    now: Date = new Date()
  ): Promise<DailySignalsCorrectionReceipt> {
    const identity = await this.resolveIdentity(userId, dto, now);
    const requestedSignals = normalizeSignals(dto.signals, false);
    const requestedHash = structuredPayloadHash(requestedSignals);
    let chain = await this.loadChain(identity);
    if (!chain) {
      throw new NotFoundException('No Daily Signals check-in exists for this pet and local day');
    }

    let tip = chain.events.at(-1)!;

    if (tip.id !== dto.expectedCurrentCareEventId) {
      if (
        tip.eventType === DAILY_SIGNALS_CORRECTION_POLICY_V1.eventType &&
        tip.correctsCareEventId === dto.expectedCurrentCareEventId &&
        tip.payloadHash === requestedHash
      ) {
        const reconciliation = await this.reconcileChain(userId, chain);
        return {
          correctionCareEventId: tip.id,
          duplicate: true,
          currentAdvanced: false,
          state: this.toEffectiveState(chain),
          ...reconciliation,
        };
      }
      throw new ConflictException(
        'Daily Signals changed after this correction screen opened. Reload before correcting.'
      );
    }

    const nextSequence = tip.sequence + 1;
    const rootId = chain.root.id;
    const dedupeKey = `daily-signals-correction:${rootId}:v${nextSequence}`;

    const reward = await this.careEvents.record({
      userId,
      petId: identity.petId,
      eventType: DAILY_SIGNALS_CORRECTION_POLICY_V1.eventType,
      pathway: 'CARE',
      occurredAt: now,
      source: DAILY_SIGNALS_CORRECTION_POLICY_V1.source,
      evidenceType: 'SELF_REPORT',
      evidenceConfidence: DAILY_SIGNALS_CORRECTION_POLICY_V1.evidenceConfidence,
      dedupeKey,
      dedupeScope: DAILY_SIGNALS_CORRECTION_POLICY_V1.dedupeScope,
      visibility: DAILY_SIGNALS_CORRECTION_POLICY_V1.visibility,
      safetyEligible: false,
      context: {
        rootCareEventId: rootId,
        correctsCareEventId: tip.id,
        correctionSequence: nextSequence,
        householdId: identity.householdId,
        localDate: identity.localDate,
        timezone: identity.timezone,
        correctionPolicyVersion: DAILY_SIGNALS_CORRECTION_POLICY_V1.version,
        payloadHash: requestedHash,
      },
      outcome: { signals: requestedSignals },
    });

    const canonical = await this.careEvents.getAuthorizedEvent(userId, reward.careEventId);
    const canonicalContext = canonical.context;
    const canonicalSignals = normalizeSignals(canonical.outcome.signals, false);
    const canonicalHash = structuredPayloadHash(canonicalSignals);
    if (
      canonical.eventType !== DAILY_SIGNALS_CORRECTION_POLICY_V1.eventType ||
      canonical.source !== DAILY_SIGNALS_CORRECTION_POLICY_V1.source ||
      canonical.evidenceType !== 'SELF_REPORT' ||
      canonical.petId !== identity.petId ||
      canonicalContext.rootCareEventId !== rootId ||
      canonicalContext.correctsCareEventId !== tip.id ||
      canonicalContext.correctionSequence !== nextSequence ||
      canonicalContext.householdId !== identity.householdId ||
      canonicalContext.localDate !== identity.localDate ||
      canonicalContext.timezone !== identity.timezone ||
      canonicalContext.correctionPolicyVersion !== DAILY_SIGNALS_CORRECTION_POLICY_V1.version ||
      canonicalContext.payloadHash !== requestedHash ||
      canonicalHash !== requestedHash
    ) {
      throw new ConflictException(
        'Concurrent Daily Signals correction used different or invalid semantics'
      );
    }

    chain = await this.loadChain(identity);
    if (!chain || !chain.events.some((event) => event.id === canonical.id)) {
      throw new ConflictException('Canonical Daily Signals correction chain could not be reloaded');
    }

    tip = chain.events.at(-1)!;
    const reconciliation = await this.reconcileChain(userId, chain);
    return {
      correctionCareEventId: canonical.id,
      duplicate: reward.duplicate,
      currentAdvanced: tip.id !== canonical.id,
      state: this.toEffectiveState(chain),
      ...reconciliation,
    };
  }

  private async resolveIdentity(
    userId: string,
    input: Pick<DailySignalsCurrentQueryDto, 'householdId' | 'petId'>,
    now: Date
  ) {
    const household = await this.households.assertHouseholdPetAccessible(
      userId,
      input.householdId,
      input.petId
    );
    if (!household.timezone) {
      throw new BadRequestException(
        'Household timezone is required before Daily Signals can be corrected'
      );
    }

    let localDate: string;
    try {
      localDate = localDateInTimeZone(now, household.timezone);
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : 'Invalid household timezone'
      );
    }

    return {
      householdId: household.householdId,
      petId: input.petId,
      timezone: household.timezone,
      localDate,
    };
  }

  private async loadChain(identity: {
    householdId: string;
    petId: string;
    timezone: string;
    localDate: string;
  }): Promise<DailySignalsChain | null> {
    const rows = await this.prisma.$queryRaw<DailySignalsEventRow[]>(Prisma.sql`
      SELECT
        id, user_id, pet_id, event_type, pathway, occurred_at, source,
        evidence_type, evidence_confidence, context, outcome, dedupe_key,
        visibility, created_at
      FROM care_events
      WHERE pet_id = ${identity.petId}
        AND source = 'INTELLIGENCE'
        AND event_type IN ('DAILY_SIGNALS_CHECKIN', 'DAILY_SIGNALS_CORRECTION')
        AND context->>'householdId' = ${identity.householdId}
        AND context->>'localDate' = ${identity.localDate}
      ORDER BY created_at ASC, id ASC
    `);

    if (rows.length === 0) return null;

    const roots = rows.filter((row) => row.event_type === 'DAILY_SIGNALS_CHECKIN');
    if (roots.length !== 1) {
      throw new ConflictException('Daily Signals local-day identity has an invalid root count');
    }
    const rootRow = roots[0]!;
    if (
      !rootRow.pet_id ||
      rootRow.source !== 'INTELLIGENCE' ||
      rootRow.evidence_type !== 'SELF_REPORT'
    ) {
      throw new ConflictException('Canonical Daily Signals root identity is invalid');
    }

    const rootSignals = normalizeSignals(rootRow.outcome?.signals, true);
    const root: ChainEvent = {
      id: rootRow.id,
      userId: rootRow.user_id,
      petId: rootRow.pet_id,
      eventType: 'DAILY_SIGNALS_CHECKIN',
      occurredAt: rootRow.occurred_at.toISOString(),
      evidenceConfidence: rootRow.evidence_confidence,
      sequence: 0,
      correctsCareEventId: null,
      payloadHash: null,
      signals: rootSignals,
    };

    const corrections = rows
      .filter((row) => row.event_type === DAILY_SIGNALS_CORRECTION_POLICY_V1.eventType)
      .map((row) => {
        if (!row.pet_id || row.outcome?.note !== undefined) {
          throw new ConflictException(
            'Canonical Daily Signals correction privacy shape is invalid'
          );
        }
        const context = row.context ?? {};
        const sequence = integerContext(context, 'correctionSequence');
        const rootCareEventId = stringContext(context, 'rootCareEventId');
        const correctsCareEventId = stringContext(context, 'correctsCareEventId');
        const policyVersion = stringContext(context, 'correctionPolicyVersion');
        const payloadHash = stringContext(context, 'payloadHash');
        const signals = normalizeSignals(row.outcome?.signals, false);

        if (
          sequence === null ||
          sequence < 1 ||
          rootCareEventId !== root.id ||
          !correctsCareEventId ||
          policyVersion !== DAILY_SIGNALS_CORRECTION_POLICY_V1.version ||
          !payloadHash ||
          payloadHash !== structuredPayloadHash(signals) ||
          row.source !== DAILY_SIGNALS_CORRECTION_POLICY_V1.source ||
          row.evidence_type !== 'SELF_REPORT'
        ) {
          throw new ConflictException('Canonical Daily Signals correction chain is malformed');
        }

        return {
          id: row.id,
          userId: row.user_id,
          petId: row.pet_id,
          eventType: DAILY_SIGNALS_CORRECTION_POLICY_V1.eventType,
          occurredAt: row.occurred_at.toISOString(),
          evidenceConfidence: row.evidence_confidence,
          sequence,
          correctsCareEventId,
          payloadHash,
          signals,
        } satisfies ChainEvent;
      })
      .sort((left, right) => left.sequence - right.sequence || left.id.localeCompare(right.id));

    let predecessor = root.id;
    for (let index = 0; index < corrections.length; index += 1) {
      const correction = corrections[index]!;
      if (correction.sequence !== index + 1 || correction.correctsCareEventId !== predecessor) {
        throw new ConflictException('Canonical Daily Signals correction sequence is not linear');
      }
      predecessor = correction.id;
    }

    return {
      ...identity,
      root,
      events: [root, ...corrections],
    };
  }

  private toEffectiveState(chain: DailySignalsChain): DailySignalsEffectiveState {
    const tip = chain.events.at(-1)!;
    return {
      rootCareEventId: chain.root.id,
      currentCareEventId: tip.id,
      correctionSequence: tip.sequence,
      householdId: chain.householdId,
      petId: chain.petId,
      localDate: chain.localDate,
      timezone: chain.timezone,
      status: tip.sequence === 0 ? 'ORIGINAL' : 'CORRECTED',
      signals: { ...tip.signals },
    };
  }

  private async reconcileChain(userId: string, chain: DailySignalsChain) {
    await this.dailySignals.replay(userId, chain.root.id);

    const eventIds = chain.events.map((event) => event.id);
    const rows = await this.prisma.$queryRaw<ProjectionRow[]>(Prisma.sql`
      SELECT id, source_event_id, dimension, retracted_at
      FROM dogos_intelligence.observations
      WHERE pet_id = ${chain.petId}
        AND source_type = 'OWNER_CHECKIN'
        AND source_event_id IN (${Prisma.join(eventIds)})
    `);

    const byEvent = new Map<string, Map<SignalDimension, ProjectionRow>>();
    for (const row of rows) {
      if (!row.source_event_id) continue;
      const eventMap =
        byEvent.get(row.source_event_id) ?? new Map<SignalDimension, ProjectionRow>();
      eventMap.set(row.dimension, row);
      byEvent.set(row.source_event_id, eventMap);
    }

    const current = new Map<SignalDimension, ProjectionRow>();
    for (const field of DAILY_SIGNAL_FIELDS) {
      const choice = chain.root.signals[field];
      if (choice === undefined || choice === 'UNSURE') continue;
      const dimension = DAILY_SIGNAL_DIMENSION_BY_FIELD[field];
      const row = byEvent.get(chain.root.id)?.get(dimension);
      if (!row) {
        throw new ConflictException(
          'Canonical Daily Signals root projection could not be repaired'
        );
      }
      current.set(dimension, row);
    }

    const projected = new Set<SignalDimension>();
    const retracted = new Set<SignalDimension>();

    for (const correction of chain.events.slice(1)) {
      for (const field of DAILY_SIGNAL_FIELDS) {
        const dimension = DAILY_SIGNAL_DIMENSION_BY_FIELD[field];
        const choice = correction.signals[field];
        const predecessor = current.get(dimension);
        const existing = byEvent.get(correction.id)?.get(dimension);

        if (choice !== undefined && choice !== 'UNSURE') {
          const candidate = normalizeOwnerCheckinObservation({
            userId: correction.userId,
            petId: chain.petId,
            careEventId: correction.id,
            dimension,
            choice,
            observedAt: chain.root.occurredAt,
            localDate: chain.localDate,
            confidence: correction.evidenceConfidence,
            ...(predecessor ? { supersedesObservationId: predecessor.id } : {}),
          });
          if (!candidate) {
            throw new ConflictException('Daily Signals correction normalization failed');
          }
          const receipt = await this.projection.projectObservation(candidate);
          const row: ProjectionRow = existing ?? {
            id: receipt.observationId,
            source_event_id: correction.id,
            dimension,
            retracted_at: null,
          };
          if (row.id !== receipt.observationId) {
            throw new ConflictException('Daily Signals correction projection identity drifted');
          }
          const eventMap = byEvent.get(correction.id) ?? new Map<SignalDimension, ProjectionRow>();
          eventMap.set(dimension, row);
          byEvent.set(correction.id, eventMap);
          current.set(dimension, row);
          projected.add(dimension);
          continue;
        }

        if (existing) {
          throw new ConflictException(
            'Uncertain or cleared Daily Signals correction has baseline projection evidence'
          );
        }
        if (predecessor) {
          await this.projection.retractObservation({
            userId,
            petId: chain.petId,
            observationId: predecessor.id,
            reason: `Daily Signals correction ${correction.id} cleared ${dimension}`,
          });
          current.delete(dimension);
          retracted.add(dimension);
        }
      }
    }

    return {
      projectedDimensions: [...projected].sort(),
      retractedDimensions: [...retracted].sort(),
    };
  }
}
