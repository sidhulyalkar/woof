import {
  BadRequestException,
  ConflictException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { Prisma } from '@woof/database';
import { createHash, randomUUID } from 'node:crypto';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { BASELINE_POLICY_V1 } from './baseline-policy-v1.receipt';
import type { NormalizedObservation, SignalDimension } from './baseline-policy-v1.types';
import { EVIDENCE_NORMALIZATION_V1 } from './evidence-normalization-v1.receipt';
import type {
  IntelligenceDimension,
  PersistedProjectionObservation,
  ProjectionObservationCandidate,
  ProjectionRetractionReceipt,
  ProjectionWriteReceipt,
  QualifiedProjectionSourceType,
} from './evidence-projection-v1.types';
import { toBaselineObservation } from './evidence-projection-v1.types';

const DAY_MS = 86_400_000;
const LOCAL_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

type ObservationRow = {
  id: string;
  user_id: string;
  pet_id: string;
  dimension: IntelligenceDimension;
  source_type: QualifiedProjectionSourceType;
  source_event_id: string | null;
  source_record_id: string | null;
  source_identity: string;
  observed_at: Date;
  ingested_at: Date;
  local_date: Date | string;
  delta_bucket: number | null;
  numeric_value: number | null;
  unit: string | null;
  confidence: number;
  reliability: 'WEAK' | 'STANDARD' | 'STRONG';
  authority: 'BASELINE_ELIGIBLE' | 'CONTEXT_ONLY';
  normalization_version: 'evidence-normalization-v1';
  normalization_reason: string;
  payload_hash: string;
  context: Record<string, unknown> | null;
  supersedes_observation_id: string | null;
  retracted_at: Date | null;
  retraction_reason: string | null;
};

type ExistingIdentityRow = {
  id: string;
  payload_hash: string;
};

type SupersessionRow = {
  id: string;
  pet_id: string;
  dimension: IntelligenceDimension;
  retracted_at: Date | null;
};

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, stableValue(entry)])
    );
  }
  return value;
}

function canonicalPayload(candidate: ProjectionObservationCandidate) {
  return {
    userId: candidate.userId,
    petId: candidate.petId,
    dimension: candidate.dimension,
    sourceType: candidate.sourceType,
    sourceIdentity: candidate.sourceIdentity,
    sourceEventId: candidate.sourceEventId ?? null,
    sourceRecordId: candidate.sourceRecordId ?? null,
    observedAt: candidate.observedAt,
    localDate: candidate.localDate,
    deltaBucket: candidate.deltaBucket ?? null,
    numericValue: candidate.numericValue ?? null,
    unit: candidate.unit ?? null,
    confidence: candidate.confidence,
    reliability: candidate.reliability,
    authority: candidate.authority,
    normalizationVersion: candidate.normalizationVersion,
    normalizationReason: candidate.normalizationReason,
    context: stableValue(candidate.context ?? {}),
    supersedesObservationId: candidate.supersedesObservationId ?? null,
  };
}

function payloadHash(candidate: ProjectionObservationCandidate) {
  return createHash('sha256')
    .update(JSON.stringify(canonicalPayload(candidate)))
    .digest('hex');
}

function localDateString(value: Date | string) {
  return typeof value === 'string' ? value.slice(0, 10) : value.toISOString().slice(0, 10);
}

@Injectable()
export class IntelligenceProjectionService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly households: HouseholdsService
  ) {}

  async projectObservation(
    candidate: ProjectionObservationCandidate
  ): Promise<ProjectionWriteReceipt> {
    this.validateCandidate(candidate);
    await this.households.assertPetAccessible(candidate.userId, candidate.petId);
    const hash = payloadHash(candidate);
    const contextJson = JSON.stringify(stableValue(candidate.context ?? {}));

    return this.prisma.$transaction(async (tx) => {
      const existing = await this.findByIdentity(tx, candidate);
      if (existing) return this.duplicateReceipt(existing, hash);

      if (candidate.supersedesObservationId) {
        const predecessor = await tx.$queryRaw<SupersessionRow[]>(Prisma.sql`
          SELECT id, pet_id, dimension, retracted_at
          FROM dogos_intelligence.observations
          WHERE id = ${candidate.supersedesObservationId}
          FOR UPDATE
        `);
        const prior = predecessor[0];
        if (!prior) throw new NotFoundException('Superseded observation not found');
        if (prior.retracted_at) {
          throw new ConflictException('Retracted evidence cannot receive a new correction');
        }
        if (prior.pet_id !== candidate.petId || prior.dimension !== candidate.dimension) {
          throw new BadRequestException('Correction must preserve pet and dimension identity');
        }

        const activeSuccessor = await tx.$queryRaw<Array<{ id: string }>>(Prisma.sql`
          SELECT id
          FROM dogos_intelligence.observations
          WHERE supersedes_observation_id = ${candidate.supersedesObservationId}
            AND retracted_at IS NULL
          LIMIT 1
        `);
        if (activeSuccessor[0]) {
          throw new ConflictException('Observation already has an active correction');
        }
      }

      const observationId = randomUUID();
      const inserted = await tx.$queryRaw<Array<{ id: string }>>(Prisma.sql`
        INSERT INTO dogos_intelligence.observations (
          id,
          user_id,
          pet_id,
          dimension,
          source_type,
          source_event_id,
          source_record_id,
          source_identity,
          observed_at,
          local_date,
          delta_bucket,
          numeric_value,
          unit,
          confidence,
          reliability,
          authority,
          normalization_version,
          normalization_reason,
          payload_hash,
          context,
          supersedes_observation_id
        ) VALUES (
          ${observationId},
          ${candidate.userId},
          ${candidate.petId},
          ${candidate.dimension},
          ${candidate.sourceType},
          ${candidate.sourceEventId ?? null},
          ${candidate.sourceRecordId ?? null},
          ${candidate.sourceIdentity},
          ${new Date(candidate.observedAt)},
          CAST(${candidate.localDate} AS DATE),
          ${candidate.deltaBucket ?? null},
          ${candidate.numericValue ?? null},
          ${candidate.unit ?? null},
          ${candidate.confidence},
          ${candidate.reliability},
          ${candidate.authority},
          ${candidate.normalizationVersion},
          ${candidate.normalizationReason},
          ${hash},
          CAST(${contextJson} AS JSONB),
          ${candidate.supersedesObservationId ?? null}
        )
        ON CONFLICT (
          pet_id,
          dimension,
          source_type,
          source_identity,
          normalization_version
        ) DO NOTHING
        RETURNING id
      `);

      if (inserted[0]) {
        return { observationId: inserted[0].id, payloadHash: hash, duplicate: false };
      }

      const raced = await this.findByIdentity(tx, candidate);
      if (!raced) {
        throw new ConflictException('Projection write conflicted with another active correction');
      }
      return this.duplicateReceipt(raced, hash);
    });
  }

  async getBaselineEvidence(input: {
    userId: string;
    petId: string;
    dimension: SignalDimension;
    now: string;
  }): Promise<NormalizedObservation[]> {
    await this.households.assertPetAccessible(input.userId, input.petId);
    const nowMs = Date.parse(input.now);
    if (!Number.isFinite(nowMs)) throw new BadRequestException('A valid explicit now is required');

    const from = new Date(nowMs - BASELINE_POLICY_V1.retentionWindowDays * DAY_MS);
    const to = new Date(nowMs);
    const rows = await this.prisma.$queryRaw<ObservationRow[]>(Prisma.sql`
      SELECT
        o.id,
        o.user_id,
        o.pet_id,
        o.dimension,
        o.source_type,
        o.source_event_id,
        o.source_record_id,
        o.source_identity,
        o.observed_at,
        o.ingested_at,
        o.local_date,
        o.delta_bucket,
        o.numeric_value,
        o.unit,
        o.confidence,
        o.reliability,
        o.authority,
        o.normalization_version,
        o.normalization_reason,
        o.payload_hash,
        o.context,
        o.supersedes_observation_id,
        o.retracted_at,
        o.retraction_reason
      FROM dogos_intelligence.observations o
      WHERE o.pet_id = ${input.petId}
        AND o.dimension = ${input.dimension}
        AND o.authority = 'BASELINE_ELIGIBLE'
        AND o.delta_bucket IS NOT NULL
        AND o.observed_at >= ${from}
        AND o.observed_at <= ${to}
        AND o.retracted_at IS NULL
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_intelligence.observations successor
          WHERE successor.supersedes_observation_id = o.id
        )
      ORDER BY o.observed_at ASC, o.id ASC
      LIMIT 512
    `);

    return rows.map((row) =>
      toBaselineObservation({
        id: row.id,
        sourceIdentity: row.source_identity,
        dimension: row.dimension,
        sourceType: row.source_type,
        observedAt: row.observed_at.toISOString(),
        localDate: localDateString(row.local_date),
        deltaBucket: row.delta_bucket as -2 | -1 | 0 | 1 | 2,
        confidence: row.confidence,
        reliability: row.reliability,
      })
    );
  }

  async retractObservation(input: {
    userId: string;
    petId: string;
    observationId: string;
    reason: string;
  }): Promise<ProjectionRetractionReceipt> {
    await this.households.assertPetAccessible(input.userId, input.petId);
    const reason = input.reason.trim();
    if (!reason || reason.length > 256) {
      throw new BadRequestException('Retraction reason must be between 1 and 256 characters');
    }

    return this.prisma.$transaction(async (tx) => {
      const rows = await tx.$queryRaw<Array<{ id: string; retracted_at: Date | null }>>(Prisma.sql`
        SELECT id, retracted_at
        FROM dogos_intelligence.observations
        WHERE id = ${input.observationId}
          AND pet_id = ${input.petId}
        FOR UPDATE
      `);
      const row = rows[0];
      if (!row) throw new NotFoundException('Observation not found');
      if (row.retracted_at) {
        return { observationId: row.id, retracted: true, duplicate: true };
      }

      await tx.$executeRaw(Prisma.sql`
        UPDATE dogos_intelligence.observations
        SET retracted_at = NOW(), retraction_reason = ${reason}
        WHERE id = ${row.id}
      `);
      return { observationId: row.id, retracted: true, duplicate: false };
    });
  }

  async getEffectiveProjectionHistory(input: {
    userId: string;
    petId: string;
    dimension: IntelligenceDimension;
    from: string;
    to: string;
    limit?: number;
  }): Promise<PersistedProjectionObservation[]> {
    await this.households.assertPetAccessible(input.userId, input.petId);
    const fromMs = Date.parse(input.from);
    const toMs = Date.parse(input.to);
    if (!Number.isFinite(fromMs) || !Number.isFinite(toMs) || fromMs > toMs) {
      throw new BadRequestException('A valid bounded projection history window is required');
    }
    if (toMs - fromMs > BASELINE_POLICY_V1.retentionWindowDays * DAY_MS) {
      throw new BadRequestException(
        'Projection history window exceeds the retained intelligence horizon'
      );
    }
    const limit = Math.max(1, Math.min(input.limit ?? 200, 512));

    const rows = await this.prisma.$queryRaw<ObservationRow[]>(Prisma.sql`
      SELECT
        o.id,
        o.user_id,
        o.pet_id,
        o.dimension,
        o.source_type,
        o.source_event_id,
        o.source_record_id,
        o.source_identity,
        o.observed_at,
        o.ingested_at,
        o.local_date,
        o.delta_bucket,
        o.numeric_value,
        o.unit,
        o.confidence,
        o.reliability,
        o.authority,
        o.normalization_version,
        o.normalization_reason,
        o.payload_hash,
        o.context,
        o.supersedes_observation_id,
        o.retracted_at,
        o.retraction_reason
      FROM dogos_intelligence.observations o
      WHERE o.pet_id = ${input.petId}
        AND o.dimension = ${input.dimension}
        AND o.observed_at >= ${new Date(fromMs)}
        AND o.observed_at <= ${new Date(toMs)}
        AND o.retracted_at IS NULL
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_intelligence.observations successor
          WHERE successor.supersedes_observation_id = o.id
        )
      ORDER BY o.observed_at ASC, o.id ASC
      LIMIT ${limit}
    `);

    return rows.map((row) => this.toPersisted(row));
  }

  private async findByIdentity(
    tx: Prisma.TransactionClient,
    candidate: ProjectionObservationCandidate
  ): Promise<ExistingIdentityRow | null> {
    const rows = await tx.$queryRaw<ExistingIdentityRow[]>(Prisma.sql`
      SELECT id, payload_hash
      FROM dogos_intelligence.observations
      WHERE pet_id = ${candidate.petId}
        AND dimension = ${candidate.dimension}
        AND source_type = ${candidate.sourceType}
        AND source_identity = ${candidate.sourceIdentity}
        AND normalization_version = ${candidate.normalizationVersion}
      LIMIT 1
    `);
    return rows[0] ?? null;
  }

  private duplicateReceipt(
    existing: ExistingIdentityRow,
    expectedHash: string
  ): ProjectionWriteReceipt {
    if (existing.payload_hash !== expectedHash) {
      throw new ConflictException(
        'Projection source identity was replayed with different semantics'
      );
    }
    return { observationId: existing.id, payloadHash: expectedHash, duplicate: true };
  }

  private validateCandidate(candidate: ProjectionObservationCandidate) {
    if (!candidate.userId || !candidate.petId) {
      throw new BadRequestException('Projection evidence requires user and pet IDs');
    }
    if (
      !candidate.sourceIdentity.trim() ||
      candidate.sourceIdentity.length > EVIDENCE_NORMALIZATION_V1.maxSourceIdentityLength
    ) {
      throw new BadRequestException('Projection source identity is invalid');
    }
    if (!Number.isFinite(Date.parse(candidate.observedAt))) {
      throw new BadRequestException('Projection observedAt is invalid');
    }
    if (!LOCAL_DATE_PATTERN.test(candidate.localDate)) {
      throw new BadRequestException('Projection localDate must already be normalized upstream');
    }
    if (
      !Number.isFinite(candidate.confidence) ||
      candidate.confidence <= 0 ||
      candidate.confidence > 1
    ) {
      throw new BadRequestException('Projection confidence must be in (0, 1]');
    }
    if (candidate.normalizationVersion !== EVIDENCE_NORMALIZATION_V1.version) {
      throw new BadRequestException('Unsupported evidence normalization version');
    }
    if (
      !candidate.normalizationReason ||
      candidate.normalizationReason.length > EVIDENCE_NORMALIZATION_V1.maxNormalizationReasonLength
    ) {
      throw new BadRequestException('Projection normalization reason is invalid');
    }
    const contextJson = JSON.stringify(stableValue(candidate.context ?? {}));
    if (Buffer.byteLength(contextJson, 'utf8') > EVIDENCE_NORMALIZATION_V1.maxContextBytes) {
      throw new BadRequestException('Projection context exceeds the privacy-thin size limit');
    }
    if (
      candidate.deltaBucket !== undefined &&
      (!Number.isInteger(candidate.deltaBucket) ||
        candidate.deltaBucket < -2 ||
        candidate.deltaBucket > 2)
    ) {
      throw new BadRequestException('Projection delta bucket must be an integer from -2 through 2');
    }
    if (
      candidate.numericValue !== undefined &&
      (!Number.isFinite(candidate.numericValue) || Number.isNaN(candidate.numericValue))
    ) {
      throw new BadRequestException('Projection numeric value must be finite');
    }
    if (candidate.deltaBucket === undefined && candidate.numericValue === undefined) {
      throw new BadRequestException(
        'Projection evidence requires a bounded bucket or numeric measurement'
      );
    }
    if (candidate.numericValue !== undefined && !candidate.unit) {
      throw new BadRequestException('Projection numeric measurements require an explicit unit');
    }
    if (candidate.unit && candidate.numericValue === undefined) {
      throw new BadRequestException('Projection units are only valid for numeric measurements');
    }

    const baselineDimension = EVIDENCE_NORMALIZATION_V1.baselineDimensions.includes(
      candidate.dimension as (typeof EVIDENCE_NORMALIZATION_V1.baselineDimensions)[number]
    );
    if (candidate.sourceType === 'OWNER_CHECKIN') {
      if (
        candidate.authority !== 'BASELINE_ELIGIBLE' ||
        !baselineDimension ||
        candidate.deltaBucket === undefined ||
        candidate.numericValue !== undefined
      ) {
        throw new BadRequestException(
          'Owner check-ins are restricted to baseline-eligible semantic evidence in evidence-normalization-v1'
        );
      }
    }
    if (candidate.authority === 'BASELINE_ELIGIBLE') {
      if (
        !baselineDimension ||
        candidate.deltaBucket === undefined ||
        candidate.sourceType !== 'OWNER_CHECKIN'
      ) {
        throw new BadRequestException(
          'Release 1 baseline authority is restricted to owner check-ins on qualified baseline dimensions'
        );
      }
    }
    if (candidate.sourceType === 'ACTIVITY') {
      if (
        candidate.authority !== 'CONTEXT_ONLY' ||
        !['ACTIVITY_LOAD', 'RECOVERY_REST_PROXY'].includes(candidate.dimension) ||
        candidate.numericValue === undefined
      ) {
        throw new BadRequestException(
          'Activity evidence is context-only in evidence-normalization-v1'
        );
      }
    }
    if (candidate.sourceType === 'COACHING') {
      if (
        candidate.authority !== 'CONTEXT_ONLY' ||
        candidate.dimension !== 'TRAINING_COMFORT_SUCCESS' ||
        candidate.numericValue === undefined
      ) {
        throw new BadRequestException(
          'Coaching evidence is context-only in evidence-normalization-v1'
        );
      }
    }
  }

  private toPersisted(row: ObservationRow): PersistedProjectionObservation {
    return {
      id: row.id,
      userId: row.user_id,
      petId: row.pet_id,
      dimension: row.dimension,
      sourceType: row.source_type,
      sourceIdentity: row.source_identity,
      ...(row.source_event_id ? { sourceEventId: row.source_event_id } : {}),
      ...(row.source_record_id ? { sourceRecordId: row.source_record_id } : {}),
      observedAt: row.observed_at.toISOString(),
      localDate: localDateString(row.local_date),
      ...(row.delta_bucket === null
        ? {}
        : { deltaBucket: row.delta_bucket as -2 | -1 | 0 | 1 | 2 }),
      ...(row.numeric_value === null ? {} : { numericValue: row.numeric_value }),
      ...(row.unit ? { unit: row.unit } : {}),
      confidence: row.confidence,
      reliability: row.reliability,
      authority: row.authority,
      normalizationVersion: row.normalization_version,
      normalizationReason: row.normalization_reason,
      context: row.context ?? {},
      ...(row.supersedes_observation_id
        ? { supersedesObservationId: row.supersedes_observation_id }
        : {}),
      ingestedAt: row.ingested_at.toISOString(),
      payloadHash: row.payload_hash,
      retractedAt: row.retracted_at?.toISOString() ?? null,
      retractionReason: row.retraction_reason,
    };
  }
}
