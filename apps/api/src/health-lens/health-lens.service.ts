import {
  BadRequestException,
  ForbiddenException,
  Injectable,
  NotFoundException,
  ServiceUnavailableException,
} from '@nestjs/common';
import { Prisma } from '@woof/database';
import * as crypto from 'crypto';
import { PrismaService } from '../prisma/prisma.service';
import { AnalyzePetHealthDto, FollowUpHealthDto } from './dto/health-lens.dto';
import { HealthAiService, type PetHealthModelResult } from './health-ai.service';
import { buildTriageText, screenEmergencyText } from './health-triage';

const HEALTH_SOURCE = 'HEALTH_LENS';
const ASSESSMENT_EVENT = 'HEALTH_ASSESSMENT';
const FOLLOW_UP_EVENT = 'HEALTH_FOLLOW_UP';
const ASSESSMENT_VERSION = 'pet-health-lens-v1';
const DAY_MS = 24 * 60 * 60 * 1000;

@Injectable()
export class HealthLensService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly ai: HealthAiService
  ) {}

  async analyze(userId: string, dto: AnalyzePetHealthDto, image?: Express.Multer.File) {
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

    const triageText = buildTriageText(dto);
    const deterministic = screenEmergencyText(triageText);
    const imageFingerprint = image
      ? crypto.createHash('sha256').update(image.buffer).digest('hex')
      : null;

    let result: PetHealthModelResult;
    let provenance:
      'deterministic-emergency-screen' | 'multimodal-model' | 'rules-only-unavailable';

    if (deterministic) {
      result = {
        triage: deterministic.level,
        confidence: 1,
        summary: deterministic.summary,
        visibleFindings: [],
        possibleCategories: [],
        photoFeedback: {
          usable: false,
          reason: 'Emergency warning signs take priority over waiting for photo interpretation.',
          betterPhotoInstructions: [],
        },
        questions: [],
        ownerActions: [deterministic.action],
        avoid: [
          'Do not delay veterinary care to continue photographing or chatting.',
          'Do not give human medication or induce vomiting unless a veterinarian or poison specialist directs you.',
        ],
        vetHandoff: {
          recommended: true,
          timing: 'now',
          summary: `Emergency warning signs reported for ${pet.name}: ${deterministic.matchedSignals.join(', ')}.`,
          bring: [
            'medication list if available',
            'relevant packaging if toxin exposure is possible',
          ],
        },
      };
      provenance = 'deterministic-emergency-screen';
    } else if (!this.ai.isConfigured()) {
      result = this.unavailableAssessment(Boolean(image));
      provenance = 'rules-only-unavailable';
    } else {
      const [recentActivities, prior] = await Promise.all([
        this.prisma.activity.findMany({
          where: {
            userId,
            petId: pet.id,
            startedAt: { gte: new Date(Date.now() - 14 * DAY_MS) },
          },
          orderBy: { startedAt: 'desc' },
          take: 12,
          select: { type: true, startedAt: true },
        }),
        this.prisma.telemetry.findMany({
          where: {
            userId,
            petId: pet.id,
            source: HEALTH_SOURCE,
            event: ASSESSMENT_EVENT,
          },
          orderBy: { createdAt: 'desc' },
          take: 6,
          select: { createdAt: true, data: true },
        }),
      ]);

      result = await this.ai.analyze({
        pet: {
          name: pet.name,
          species: pet.species,
          breed: pet.breed,
          ageYears: this.ageYears(pet.birthdate),
          temperament: pet.temperament,
        },
        concern: dto.concern,
        bodyArea: dto.bodyArea,
        onset: dto.onset,
        appetite: dto.appetite,
        energy: dto.energy,
        breathing: dto.breathing,
        bathroom: dto.bathroom,
        recentContext: recentActivities.map((activity) => ({
          type: activity.type,
          startedAt: activity.startedAt.toISOString(),
        })),
        priorHealthObservations: prior.map((entry) =>
          this.priorSummary(entry.createdAt, entry.data)
        ),
        image: image ? { mimeType: image.mimetype, bytes: image.buffer } : undefined,
      });
      provenance = 'multimodal-model';
    }

    const shouldSave = dto.saveToTimeline !== false;
    const saved = shouldSave
      ? await this.persistAssessment({
          userId,
          petId: pet.id,
          dto,
          result,
          provenance,
          imageFingerprint,
          hadImage: Boolean(image),
        })
      : null;

    return {
      assessmentId: saved?.id ?? null,
      generatedAt: new Date().toISOString(),
      pet: { id: pet.id, name: pet.name, species: pet.species, breed: pet.breed },
      assessment: result,
      provenance: {
        version: ASSESSMENT_VERSION,
        pathway: provenance,
        imageAnalyzed: Boolean(image) && provenance === 'multimodal-model',
        modelConfigured: this.ai.isConfigured(),
        savedToTimeline: Boolean(saved),
      },
      privacy: {
        imageStoredByWoof: false,
        imagePolicy:
          'Image bytes are processed transiently for this assessment and are not written to Woof object storage. The timeline stores only derived assessment data and an irreversible image fingerprint for traceability.',
      },
      safety:
        'Health Lens is a screening and documentation aid, not a diagnosis or substitute for an examination by a licensed veterinarian.',
    };
  }

  async followUp(userId: string, dto: FollowUpHealthDto) {
    const previous = await this.prisma.telemetry.findFirst({
      where: {
        id: dto.assessmentId,
        userId,
        source: HEALTH_SOURCE,
        event: ASSESSMENT_EVENT,
      },
    });
    if (!previous) throw new NotFoundException('Health assessment not found');

    const previousData = this.asObject(previous.data);
    const petId = typeof previous.petId === 'string' ? previous.petId : null;
    if (!petId) throw new BadRequestException('Health assessment is missing pet context');

    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
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

    const emergency = screenEmergencyText(dto.message);
    let result: PetHealthModelResult;
    let pathway: string;

    if (emergency) {
      result = {
        triage: 'emergency_now',
        confidence: 1,
        summary: emergency.summary,
        visibleFindings: [],
        possibleCategories: [],
        photoFeedback: {
          usable: false,
          reason: 'New emergency warning signs take priority.',
          betterPhotoInstructions: [],
        },
        questions: [],
        ownerActions: [emergency.action],
        avoid: ['Do not delay veterinary care to continue the chat.'],
        vetHandoff: {
          recommended: true,
          timing: 'now',
          summary: `New emergency warning sign reported for ${pet.name}.`,
          bring: [],
        },
      };
      pathway = 'deterministic-emergency-screen';
    } else if (!this.ai.isConfigured()) {
      result = this.unavailableAssessment(false);
      pathway = 'rules-only-unavailable';
    } else {
      const priorAssessment = this.extractAssessment(previousData);
      result = await this.ai.analyze({
        pet: {
          name: pet.name,
          species: pet.species,
          breed: pet.breed,
          ageYears: this.ageYears(pet.birthdate),
          temperament: pet.temperament,
        },
        concern: `Previous assessment: ${JSON.stringify(priorAssessment)}\nOwner follow-up: ${dto.message}`,
        recentContext: [],
        priorHealthObservations: [],
      });
      pathway = 'multimodal-model-follow-up';
    }

    const saved = await this.prisma.telemetry.create({
      data: {
        source: HEALTH_SOURCE,
        event: FOLLOW_UP_EVENT,
        userId,
        petId,
        data: {
          version: ASSESSMENT_VERSION,
          assessmentId: previous.id,
          ownerMessage: dto.message,
          pathway,
          result,
        } as Prisma.InputJsonValue,
      },
      select: { id: true, createdAt: true },
    });

    return {
      followUpId: saved.id,
      assessmentId: previous.id,
      generatedAt: saved.createdAt.toISOString(),
      assessment: result,
      provenance: { version: ASSESSMENT_VERSION, pathway, imageReused: false },
      safety:
        'A follow-up chat can use the prior structured findings, but Woof does not retain the original image. Upload a new image when a visual change needs reassessment.',
    };
  }

  async timeline(userId: string, petId: string, limit = 20) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true },
    });
    if (!pet) throw new ForbiddenException('You do not have access to this pet');

    const entries = await this.prisma.telemetry.findMany({
      where: {
        userId,
        petId,
        source: HEALTH_SOURCE,
        event: { in: [ASSESSMENT_EVENT, FOLLOW_UP_EVENT] },
      },
      orderBy: { createdAt: 'desc' },
      take: Math.max(1, Math.min(50, limit)),
      select: { id: true, event: true, data: true, createdAt: true },
    });

    return entries.map((entry) => {
      const data = this.asObject(entry.data);
      const result = this.extractAssessment(data);
      return {
        id: entry.id,
        kind: entry.event === ASSESSMENT_EVENT ? 'assessment' : 'follow-up',
        createdAt: entry.createdAt.toISOString(),
        triage: result?.triage ?? 'insufficient_information',
        summary: result?.summary ?? 'Health observation',
        bodyArea: typeof data.bodyArea === 'string' ? data.bodyArea : null,
        concern: typeof data.concern === 'string' ? data.concern : null,
        hadImage: data.hadImage === true,
      };
    });
  }

  async deleteTimelineEntry(userId: string, entryId: string) {
    const entry = await this.prisma.telemetry.findFirst({
      where: { id: entryId, userId, source: HEALTH_SOURCE },
      select: { id: true },
    });
    if (!entry) throw new NotFoundException('Health timeline entry not found');
    await this.prisma.telemetry.delete({ where: { id: entry.id } });
    return { deleted: true };
  }

  private async persistAssessment(input: {
    userId: string;
    petId: string;
    dto: AnalyzePetHealthDto;
    result: PetHealthModelResult;
    provenance: string;
    imageFingerprint: string | null;
    hadImage: boolean;
  }) {
    return this.prisma.telemetry.create({
      data: {
        source: HEALTH_SOURCE,
        event: ASSESSMENT_EVENT,
        userId: input.userId,
        petId: input.petId,
        data: {
          version: ASSESSMENT_VERSION,
          concern: input.dto.concern,
          bodyArea: input.dto.bodyArea ?? null,
          onset: input.dto.onset ?? null,
          appetite: input.dto.appetite ?? null,
          energy: input.dto.energy ?? null,
          breathing: input.dto.breathing ?? null,
          bathroom: input.dto.bathroom ?? null,
          pathway: input.provenance,
          hadImage: input.hadImage,
          imageSha256: input.imageFingerprint,
          result: input.result,
        } as Prisma.InputJsonValue,
      },
      select: { id: true, createdAt: true },
    });
  }

  private unavailableAssessment(hadImage: boolean): PetHealthModelResult {
    return {
      triage: 'insufficient_information',
      confidence: 0,
      summary: hadImage
        ? 'This deployment cannot safely interpret the image because the multimodal health model is not configured.'
        : 'Woof does not have enough validated information to assess this concern in this deployment.',
      visibleFindings: [],
      possibleCategories: [],
      photoFeedback: {
        usable: false,
        reason: hadImage ? 'Image model unavailable.' : 'No image was analyzed.',
        betterPhotoInstructions: [],
      },
      questions: [],
      ownerActions: [
        'If the change is new, worsening, painful, persistent, or concerning to you, contact your veterinarian.',
        'Record when it started and any changes in eating, drinking, breathing, energy, urination, stool, walking, or behavior.',
      ],
      avoid: [
        'Do not use unverified home remedies or human medications based on an automated screening result.',
      ],
      vetHandoff: {
        recommended: true,
        timing: 'routine',
        summary:
          'Automated multimodal screening was unavailable, so veterinary review is the reliable next step if the concern persists.',
        bring: ['timeline of changes', 'clear photos or videos if safe to capture'],
      },
    };
  }

  private priorSummary(createdAt: Date, value: Prisma.JsonValue | null) {
    const data = this.asObject(value);
    const result = this.extractAssessment(data);
    return {
      createdAt: createdAt.toISOString(),
      triage: result?.triage ?? 'unknown',
      summary: result?.summary ?? 'Previous health observation',
    };
  }

  private extractAssessment(data: Record<string, unknown>): PetHealthModelResult | null {
    const candidate = data.result;
    if (!candidate || Array.isArray(candidate) || typeof candidate !== 'object') return null;
    return candidate as PetHealthModelResult;
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
