import { ForbiddenException, Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import * as crypto from 'crypto';
import { PrismaService } from '../prisma/prisma.service';
import { deriveIndividualBehaviorProfile } from './behavior-profile';
import { BehaviorVisionModelService } from './behavior-vision.model';
import {
  BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
  type BehaviorObservationContext,
  type BehaviorVisionModelAnalysis,
  type StoredBehaviorObservation,
} from './behavior-vision.types';
import { AnalyzeBehaviorMediaDto, BehaviorObservationFeedbackDto } from './dto/behavior-vision.dto';

const SOURCE = 'BEHAVIOR_VISION';
const OBSERVATION_EVENT = 'BEHAVIOR_OBSERVATION';
const FEEDBACK_EVENT = 'BEHAVIOR_OBSERVATION_FEEDBACK';
const DAY_MS = 24 * 60 * 60 * 1000;

@Injectable()
export class BehaviorVisionService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly model: BehaviorVisionModelService
  ) {}

  async analyze(userId: string, dto: AnalyzeBehaviorMediaDto, media: Express.Multer.File) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: dto.petId, ownerId: userId },
      select: {
        id: true,
        name: true,
        species: true,
        breed: true,
        birthdate: true,
        temperament: true,
      },
    });
    if (!pet) throw new ForbiddenException('You do not have access to this pet');

    const previous = await this.loadStoredObservations(userId, pet.id, 80);
    const priorProfile = deriveIndividualBehaviorProfile(pet.id, previous);
    const context: BehaviorObservationContext = {
      context: dto.context,
      sessionKey: dto.sessionKey?.trim() || undefined,
      phase: dto.phase ?? 'baseline',
      handlerAction: dto.handlerAction ?? 'none',
      leashState: dto.leashState ?? 'unknown',
      otherDogsPresent: dto.otherDogsPresent,
      otherDogDistanceMeters: dto.otherDogDistanceMeters,
      familiarDog: dto.familiarDog,
      audioAnalysisAllowed: dto.includeAudio === true,
      ownerNote: dto.ownerNote?.trim() || undefined,
    };

    let analysis: BehaviorVisionModelAnalysis;
    let pathway: 'specialized-model' | 'model-unavailable';
    if (this.model.isConfigured()) {
      analysis = await this.model.analyze({
        pet: {
          name: pet.name,
          species: pet.species,
          breed: pet.breed,
          ageYears: this.ageYears(pet.birthdate),
          temperament: pet.temperament,
        },
        context,
        question: dto.question,
        priorProfileSummary: {
          sampleCount: priorProfile.sampleCount,
          personalizationConfidence: priorProfile.personalizationConfidence,
          baselines: priorProfile.baselines.map((baseline) => ({
            dimension: baseline.dimension,
            mean: baseline.mean,
            confidence: baseline.confidence,
          })),
        },
        media: {
          mimeType: media.mimetype,
          bytes: media.buffer,
          filename: media.originalname || `behavior-${Date.now()}`,
        },
      });
      pathway = 'specialized-model';
    } else {
      analysis = this.unavailableAnalysis();
      pathway = 'model-unavailable';
    }

    const mediaSha256 = crypto.createHash('sha256').update(media.buffer).digest('hex');
    const mediaType = media.mimetype.startsWith('video/') ? 'video' : 'image';
    const shouldSave = dto.saveToTimeline !== false;

    const saved = shouldSave
      ? await this.prisma.telemetry.create({
          data: {
            source: SOURCE,
            event: OBSERVATION_EVENT,
            userId,
            petId: pet.id,
            data: {
              schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
              mediaType,
              mediaSha256,
              pathway,
              context,
              analysis,
            } as Prisma.InputJsonValue,
          },
          select: { id: true, createdAt: true },
        })
      : null;

    const current: StoredBehaviorObservation = {
      id: saved?.id ?? `transient-${mediaSha256.slice(0, 12)}`,
      petId: pet.id,
      createdAt: saved?.createdAt.toISOString() ?? new Date().toISOString(),
      mediaType,
      mediaSha256,
      context,
      analysis,
    };
    const profile = deriveIndividualBehaviorProfile(pet.id, [current, ...previous]);

    return {
      observationId: saved?.id ?? null,
      generatedAt: current.createdAt,
      pet: {
        id: pet.id,
        name: pet.name,
        species: pet.species,
        breed: pet.breed,
      },
      context,
      analysis,
      coach: this.coachResponse(analysis, profile),
      profile,
      provenance: {
        pathway,
        schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
        modelConfigured: this.model.isConfigured(),
        savedToTimeline: Boolean(saved),
      },
      privacy: {
        mediaStoredByWoof: false,
        audioAnalysisAllowed: context.audioAnalysisAllowed,
        mediaPolicy:
          'Raw behavior images and videos are processed transiently and are not written to Woof object storage. The timeline stores derived observations plus an irreversible media fingerprint. Audio analysis is disabled unless the owner explicitly opts in.',
      },
      safety:
        'Behavior Vision describes observable patterns and compares this dog with their own history. It cannot know a dog’s internal emotional state from video alone and never automatically recommends direct dog-to-dog greeting.',
    };
  }

  async profile(userId: string, petId: string) {
    await this.requireOwnedPet(userId, petId);
    const observations = await this.loadStoredObservations(userId, petId, 100);
    return deriveIndividualBehaviorProfile(petId, observations);
  }

  async timeline(userId: string, petId: string, limit = 30) {
    await this.requireOwnedPet(userId, petId);
    const observations = await this.loadStoredObservations(
      userId,
      petId,
      Math.max(1, Math.min(100, limit))
    );
    return observations;
  }

  async recordFeedback(userId: string, dto: BehaviorObservationFeedbackDto) {
    const observation = await this.prisma.telemetry.findFirst({
      where: {
        id: dto.observationId,
        userId,
        source: SOURCE,
        event: OBSERVATION_EVENT,
      },
      select: { id: true, petId: true },
    });
    if (!observation) throw new NotFoundException('Behavior observation not found');

    const feedback = await this.prisma.telemetry.create({
      data: {
        source: SOURCE,
        event: FEEDBACK_EVENT,
        userId,
        petId: observation.petId,
        data: {
          observationId: observation.id,
          accurate: dto.accurate,
          note: dto.note?.trim() || null,
        } as Prisma.InputJsonValue,
      },
      select: { id: true, createdAt: true },
    });

    return {
      feedbackId: feedback.id,
      observationId: observation.id,
      accurate: dto.accurate,
      createdAt: feedback.createdAt.toISOString(),
      learning:
        'Owner corrections are treated as higher-value personalization evidence. Rejected observations are excluded from this dog’s behavioral baseline.',
    };
  }

  async deleteObservation(userId: string, observationId: string) {
    const observation = await this.prisma.telemetry.findFirst({
      where: {
        id: observationId,
        userId,
        source: SOURCE,
        event: OBSERVATION_EVENT,
      },
      select: { id: true },
    });
    if (!observation) throw new NotFoundException('Behavior observation not found');

    const feedback = await this.prisma.telemetry.findMany({
      where: { userId, source: SOURCE, event: FEEDBACK_EVENT },
      select: { id: true, data: true },
    });
    const relatedFeedbackIds = feedback
      .filter((entry) => this.asObject(entry.data).observationId === observationId)
      .map((entry) => entry.id);

    await this.prisma.$transaction([
      ...(relatedFeedbackIds.length
        ? [this.prisma.telemetry.deleteMany({ where: { id: { in: relatedFeedbackIds } } })]
        : []),
      this.prisma.telemetry.delete({ where: { id: observation.id } }),
    ]);

    return { deleted: true };
  }

  private async loadStoredObservations(userId: string, petId: string, limit: number) {
    const since = new Date(Date.now() - 365 * DAY_MS);
    const [entries, feedbackEntries] = await Promise.all([
      this.prisma.telemetry.findMany({
        where: {
          userId,
          petId,
          source: SOURCE,
          event: OBSERVATION_EVENT,
          createdAt: { gte: since },
        },
        orderBy: { createdAt: 'desc' },
        take: limit,
        select: { id: true, petId: true, createdAt: true, data: true },
      }),
      this.prisma.telemetry.findMany({
        where: {
          userId,
          petId,
          source: SOURCE,
          event: FEEDBACK_EVENT,
          createdAt: { gte: since },
        },
        orderBy: { createdAt: 'desc' },
        take: 200,
        select: { data: true },
      }),
    ]);

    const feedbackByObservation = new Map<string, { accurate: boolean; note?: string }>();
    for (const entry of feedbackEntries) {
      const data = this.asObject(entry.data);
      if (typeof data.observationId !== 'string' || typeof data.accurate !== 'boolean') continue;
      if (feedbackByObservation.has(data.observationId)) continue;
      feedbackByObservation.set(data.observationId, {
        accurate: data.accurate,
        note: typeof data.note === 'string' ? data.note : undefined,
      });
    }

    return entries
      .map((entry) => {
        const data = this.asObject(entry.data);
        const context = data.context;
        const analysis = data.analysis;
        if (
          !context ||
          Array.isArray(context) ||
          typeof context !== 'object' ||
          !analysis ||
          Array.isArray(analysis) ||
          typeof analysis !== 'object'
        ) {
          return null;
        }

        const mediaType = data.mediaType === 'video' ? 'video' : 'image';
        const mediaSha256 = typeof data.mediaSha256 === 'string' ? data.mediaSha256 : '';
        return {
          id: entry.id,
          petId: entry.petId ?? petId,
          createdAt: entry.createdAt.toISOString(),
          mediaType,
          mediaSha256,
          context: {
            ...(context as StoredBehaviorObservation['context']),
            audioAnalysisAllowed:
              (context as Record<string, unknown>).audioAnalysisAllowed === true,
          },
          analysis: analysis as StoredBehaviorObservation['analysis'],
          ownerFeedback: feedbackByObservation.get(entry.id),
        } satisfies StoredBehaviorObservation;
      })
      .filter((entry): entry is StoredBehaviorObservation => entry !== null);
  }

  private coachResponse(
    analysis: BehaviorVisionModelAnalysis,
    profile: ReturnType<typeof deriveIndividualBehaviorProfile>
  ) {
    if (!analysis.mediaQuality.usable) {
      return {
        headline: 'Woof needs a clearer observation before coaching from this clip',
        explanation:
          analysis.mediaQuality.issues.join(', ') ||
          'The media did not provide enough reliable visual evidence.',
        nextSteps: analysis.mediaQuality.recaptureInstructions,
      };
    }

    const topHypothesis = [...analysis.hypotheses].sort(
      (left, right) => right.confidence - left.confidence
    )[0];

    return {
      headline: profile.recommendation.headline,
      explanation: profile.recommendation.explanation,
      observableSummary: analysis.observableSummary,
      hypothesis:
        topHypothesis && topHypothesis.confidence >= 0.45
          ? {
              statement: topHypothesis.statement,
              confidence: topHypothesis.confidence,
              caveat:
                'This is a behavior-compatible hypothesis, not a direct readout of emotion or intent.',
            }
          : null,
      nextSteps: profile.recommendation.nextSafeExperiment,
      socialSafety:
        'Barking, pulling, pacing, or strong orientation toward another dog can occur with excitement, frustration, fear, uncertainty, learned leash patterns, or mixed motivation. Woof does not infer “needs to greet” from those signals alone.',
    };
  }

  private unavailableAnalysis(): BehaviorVisionModelAnalysis {
    return {
      schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
      modelVersion: 'unavailable',
      featureVersion: 'unavailable',
      mediaQuality: {
        usable: false,
        confidence: 0,
        issues: ['specialized behavior-video model unavailable'],
        recaptureInstructions: [],
      },
      evidence: [],
      dimensions: [],
      hypotheses: [
        {
          id: 'insufficient-evidence',
          confidence: 1,
          statement:
            'No behavior inference was produced because the specialized model is unavailable.',
          supportingEvidence: [],
          contradictoryEvidence: [],
        },
      ],
      observableSummary: 'No automated behavior observation was generated.',
      uncertainty: 'Model unavailable; Woof did not infer behavior from the media.',
    };
  }

  private async requireOwnedPet(userId: string, petId: string) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true },
    });
    if (!pet) throw new ForbiddenException('You do not have access to this pet');
    return pet;
  }

  private asObject(value: Prisma.JsonValue | null): Record<string, unknown> {
    if (!value || Array.isArray(value) || typeof value !== 'object') return {};
    return value as Record<string, unknown>;
  }

  private ageYears(birthdate: Date | null) {
    if (!birthdate) return null;
    return Math.max(
      0,
      Math.round(((Date.now() - birthdate.getTime()) / (365.25 * DAY_MS)) * 10) / 10
    );
  }
}
