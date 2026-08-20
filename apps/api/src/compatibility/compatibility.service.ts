import {
  BadRequestException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';

type CompatibilityPet = {
  id: string;
  ownerId?: string;
  name: string;
  species: string;
  breed: string | null;
  birthdate: Date | null;
  temperament: Prisma.JsonValue | null;
};

type PairOutcome = {
  rating: number | null;
  feedbackTags: string[];
  occurredAt?: Date | null;
  createdAt?: Date;
};

type CompatibilityFactors = {
  species: number;
  temperament?: number;
  age?: number;
  breed?: number;
  outcomes?: number;
};

type BehaviorVector = {
  energy: number | null;
  sociability: number | null;
  caution: number | null;
  excitability: number | null;
  trainability: number | null;
  socialRisk: number | null;
  coverage: number;
};

const BASELINE_VERSION = 'behavior-outcome-baseline-v2';
const DAY_MS = 24 * 60 * 60 * 1000;

@Injectable()
export class CompatibilityService {
  constructor(private readonly prisma: PrismaService) {}

  async getOrCreatePetEdge(petAId: string, petBId: string) {
    if (!petAId || !petBId) {
      throw new BadRequestException('Both pet IDs are required');
    }
    if (petAId === petBId) {
      throw new BadRequestException('A pet cannot be matched with itself');
    }

    const [firstPetId, secondPetId] = [petAId, petBId].sort();
    const [petA, petB] = await Promise.all([
      this.prisma.pet.findUnique({ where: { id: firstPetId } }),
      this.prisma.pet.findUnique({ where: { id: secondPetId } }),
    ]);

    if (!petA) {
      throw new NotFoundException(`Pet with ID ${firstPetId} not found`);
    }
    if (!petB) {
      throw new NotFoundException(`Pet with ID ${secondPetId} not found`);
    }

    const existing = await this.prisma.petEdge.findUnique({
      where: {
        petAId_petBId: { petAId: firstPetId, petBId: secondPetId },
      },
      include: { petA: true, petB: true },
    });
    if (existing) return existing;

    return this.prisma.petEdge.create({
      data: {
        petAId: firstPetId,
        petBId: secondPetId,
        weight: 1,
        status: 'PROPOSED',
      },
      include: { petA: true, petB: true },
    });
  }

  async getOrCreatePetEdgeForActor(
    userId: string,
    petAId: string,
    petBId: string,
  ) {
    await this.assertActorOwnsEither(userId, petAId, petBId);
    return this.getOrCreatePetEdge(petAId, petBId);
  }

  async calculateCompatibility(
    userId: string,
    petAId: string,
    petBId: string,
  ) {
    await this.assertActorOwnsEither(userId, petAId, petBId);
    const edge = await this.getOrCreatePetEdge(petAId, petBId);
    const outcomes = await this.getPairOutcomes(edge.petA.ownerId, edge.petB.ownerId);
    const result = this.scoreBaseline(edge.petA, edge.petB, outcomes);

    const updatedEdge = await this.prisma.petEdge.update({
      where: { id: edge.id },
      data: {
        compatibilityScore: result.score,
        lastInteractionAt: new Date(),
      },
      include: {
        petA: {
          select: {
            id: true,
            name: true,
            species: true,
            breed: true,
            avatarUrl: true,
          },
        },
        petB: {
          select: {
            id: true,
            name: true,
            species: true,
            breed: true,
            avatarUrl: true,
          },
        },
      },
    });

    return {
      petAId,
      petBId,
      compatibilityScore: result.score,
      confidence: result.confidence,
      source: BASELINE_VERSION,
      factors: result.factors,
      explanation: result.explanation,
      edge: updatedEdge,
    };
  }

  async getRecommendations(userId: string, petId: string, limit = 10) {
    const safeLimit = Math.max(1, Math.min(Number(limit) || 10, 50));
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
    });
    if (!pet) {
      throw new NotFoundException('Pet not found');
    }

    const [blocked, edges, outcomes, candidates] = await Promise.all([
      this.prisma.blockedUser.findMany({
        where: { OR: [{ userId }, { blockedId: userId }] },
        select: { userId: true, blockedId: true },
      }),
      this.prisma.petEdge.findMany({
        where: { OR: [{ petAId: petId }, { petBId: petId }] },
        select: {
          id: true,
          petAId: true,
          petBId: true,
          status: true,
          lastInteractionAt: true,
        },
      }),
      this.prisma.meetupProposal.findMany({
        where: {
          OR: [{ proposerId: userId }, { recipientId: userId }],
          createdAt: { gte: new Date(Date.now() - 180 * DAY_MS) },
        },
        select: {
          proposerId: true,
          recipientId: true,
          status: true,
          occurredAt: true,
          rating: true,
          feedbackTags: true,
          createdAt: true,
        },
      }),
      this.prisma.pet.findMany({
        where: {
          id: { not: petId },
          ownerId: { not: userId },
          species: pet.species,
          owner: { visibility: { not: 'PRIVATE' } },
        },
        take: 150,
        orderBy: { updatedAt: 'desc' },
        include: {
          owner: {
            select: {
              id: true,
              handle: true,
              bio: true,
              avatarUrl: true,
              isVerified: true,
              visibility: true,
            },
          },
        },
      }),
    ]);

    const blockedOwnerIds = new Set(
      blocked.map((entry) =>
        entry.userId === userId ? entry.blockedId : entry.userId,
      ),
    );
    const edgeByPet = new Map(
      edges.map((edge) => [
        edge.petAId === petId ? edge.petBId : edge.petAId,
        edge,
      ]),
    );

    const recommendations = candidates
      .filter((candidate) => !blockedOwnerIds.has(candidate.ownerId))
      .filter((candidate) => edgeByPet.get(candidate.id)?.status !== 'AVOID')
      .map((candidate) => {
        const candidateOutcomes = outcomes
          .filter((outcome) =>
            [outcome.proposerId, outcome.recipientId].includes(candidate.ownerId),
          )
          .map<PairOutcome>((outcome) => ({
            rating: outcome.rating,
            feedbackTags: outcome.feedbackTags,
            occurredAt: outcome.occurredAt,
            createdAt: outcome.createdAt,
          }));
        const result = this.scoreBaseline(pet, candidate, candidateOutcomes);
        const edge = edgeByPet.get(candidate.id);

        return {
          id: edge?.id ?? `candidate-${candidate.id}`,
          pet: {
            id: candidate.id,
            ownerId: candidate.ownerId,
            name: candidate.name,
            species: candidate.species,
            breed: candidate.breed,
            birthdate: candidate.birthdate,
            avatarUrl: candidate.avatarUrl,
            temperament: this.temperamentTraits(candidate.temperament),
            owner: {
              id: candidate.owner.id,
              handle: candidate.owner.handle,
              bio: candidate.owner.bio,
              avatarUrl: candidate.owner.avatarUrl,
              isVerified: candidate.owner.isVerified,
            },
          },
          compatibilityScore: result.score,
          confidence: result.confidence,
          source: BASELINE_VERSION,
          factors: result.factors,
          explanation: result.explanation,
          status: edge?.status ?? 'PROPOSED',
          lastInteractionAt: edge?.lastInteractionAt ?? null,
        };
      })
      .sort((a, b) => {
        if (b.compatibilityScore !== a.compatibilityScore) {
          return b.compatibilityScore - a.compatibilityScore;
        }
        return b.confidence - a.confidence;
      })
      .slice(0, safeLimit);

    return {
      petId,
      recommendations,
      total: recommendations.length,
      source: BASELINE_VERSION,
      rankingPrinciples: [
        'individual behavior over breed stereotypes',
        'observed meetup outcomes when available',
        'confidence falls when profile evidence is sparse',
        'blocked, avoided, and private relationships are filtered before ranking',
      ],
    };
  }

  async updateEdgeStatus(
    userId: string,
    petAId: string,
    petBId: string,
    status: string,
  ) {
    const validStatuses = ['PROPOSED', 'CONFIRMED', 'AVOID'];
    if (!validStatuses.includes(status)) {
      throw new BadRequestException(
        `Invalid status. Must be one of: ${validStatuses.join(', ')}`,
      );
    }

    await this.assertActorOwnsEither(userId, petAId, petBId);
    const edge = await this.getOrCreatePetEdge(petAId, petBId);

    return this.prisma.petEdge.update({
      where: { id: edge.id },
      data: { status },
      include: {
        petA: { select: { id: true, name: true, avatarUrl: true } },
        petB: { select: { id: true, name: true, avatarUrl: true } },
      },
    });
  }

  async getAllEdges(
    userId: string,
    skip = 0,
    take = 20,
    status?: string,
  ) {
    const safeSkip = Math.max(0, Number(skip) || 0);
    const safeTake = Math.max(1, Math.min(Number(take) || 20, 100));
    const ownedPets = await this.prisma.pet.findMany({
      where: { ownerId: userId },
      select: { id: true },
    });
    const petIds = ownedPets.map((pet) => pet.id);
    if (petIds.length === 0) {
      return { edges: [], total: 0, skip: safeSkip, take: safeTake };
    }

    const where: Prisma.PetEdgeWhereInput = {
      OR: [{ petAId: { in: petIds } }, { petBId: { in: petIds } }],
      ...(status ? { status } : {}),
    };

    const [edges, total] = await Promise.all([
      this.prisma.petEdge.findMany({
        where,
        skip: safeSkip,
        take: safeTake,
        include: {
          petA: {
            select: { id: true, name: true, species: true, avatarUrl: true },
          },
          petB: {
            select: { id: true, name: true, species: true, avatarUrl: true },
          },
        },
        orderBy: { lastInteractionAt: 'desc' },
      }),
      this.prisma.petEdge.count({ where }),
    ]);

    return { edges, total, skip: safeSkip, take: safeTake };
  }

  private scoreBaseline(
    petA: CompatibilityPet,
    petB: CompatibilityPet,
    outcomes: PairOutcome[] = [],
  ) {
    const weightedFactors: Array<{
      key: keyof CompatibilityFactors;
      value: number;
      weight: number;
    }> = [];

    const sameSpecies =
      this.normalizeText(petA.species) === this.normalizeText(petB.species);
    const speciesScore = sameSpecies ? 1 : 0.1;
    weightedFactors.push({ key: 'species', value: speciesScore, weight: 0.15 });

    const behavior = this.behaviorCompatibility(
      petA.temperament,
      petB.temperament,
    );
    if (behavior.score !== null) {
      weightedFactors.push({
        key: 'temperament',
        value: behavior.score,
        weight: 0.5,
      });
    }

    const ageScore = this.ageSimilarity(petA.birthdate, petB.birthdate);
    if (ageScore !== null) {
      weightedFactors.push({ key: 'age', value: ageScore, weight: 0.15 });
    }

    const outcomeScore = this.outcomeScore(outcomes);
    if (outcomeScore !== null) {
      weightedFactors.push({
        key: 'outcomes',
        value: outcomeScore,
        weight: 0.18,
      });
    }

    if (petA.breed && petB.breed) {
      const sameBreed =
        this.normalizeText(petA.breed) === this.normalizeText(petB.breed);
      weightedFactors.push({
        key: 'breed',
        value: sameBreed ? 1 : 0.78,
        weight: 0.02,
      });
    }

    const totalWeight = weightedFactors.reduce(
      (sum, factor) => sum + factor.weight,
      0,
    );
    let score =
      weightedFactors.reduce(
        (sum, factor) => sum + factor.value * factor.weight,
        0,
      ) / totalWeight;

    if (behavior.socialRisk !== null && behavior.socialRisk > 0.72) {
      score *= 0.82;
    }

    const factors = weightedFactors.reduce<CompatibilityFactors>(
      (acc, factor) => ({ ...acc, [factor.key]: this.round(factor.value) }),
      { species: this.round(speciesScore) },
    );

    const profileCoverage = Math.min(
      1,
      0.2 +
        behavior.coverage * 0.45 +
        (ageScore !== null ? 0.15 : 0) +
        (petA.breed && petB.breed ? 0.02 : 0) +
        Math.min(0.18, outcomes.length * 0.06),
    );
    const confidence = Math.min(0.96, 0.34 + profileCoverage * 0.62);

    const explanation: string[] = [];
    if (behavior.score !== null && behavior.score >= 0.78) {
      explanation.push(
        'Their individual behavior profiles align across the traits Woof can currently observe',
      );
    } else if (behavior.score !== null && behavior.score >= 0.58) {
      explanation.push(
        'Their individual behavior profiles show useful overlap with some differences to introduce gradually',
      );
    } else if (behavior.score !== null) {
      explanation.push(
        'Their current behavior profiles differ enough that introductions should be gradual and owner-controlled',
      );
    }
    if (behavior.socialRisk !== null && behavior.socialRisk > 0.72) {
      explanation.push(
        'One profile contains stronger fear or aggression-related signals, so Woof applies a conservative social-safety penalty',
      );
    }
    if (ageScore !== null && ageScore >= 0.85) {
      explanation.push('They are relatively close in life stage');
    }
    if (outcomeScore !== null) {
      explanation.push(
        `${outcomes.length} recent owner-reported meetup outcome${outcomes.length === 1 ? '' : 's'} inform this estimate`,
      );
    }
    if (
      petA.breed &&
      petB.breed &&
      this.normalizeText(petA.breed) === this.normalizeText(petB.breed)
    ) {
      explanation.push(
        'They share a breed label, used only as a very small supporting prior',
      );
    }
    if (profileCoverage < 0.62) {
      explanation.push(
        'Confidence is intentionally limited because Woof still has sparse individual behavior or outcome evidence',
      );
    }
    if (explanation.length === 0) {
      explanation.push(
        'This is a conservative cold-start estimate that should improve with owner feedback and real interaction outcomes',
      );
    }

    return {
      score: this.round(this.clamp01(score)),
      confidence: this.round(confidence),
      factors,
      explanation,
    };
  }

  private behaviorCompatibility(
    rawA: Prisma.JsonValue | null,
    rawB: Prisma.JsonValue | null,
  ) {
    const a = this.behaviorVector(rawA);
    const b = this.behaviorVector(rawB);
    const pairs: Array<{ a: number | null; b: number | null; weight: number }> = [
      { a: a.energy, b: b.energy, weight: 0.28 },
      { a: a.sociability, b: b.sociability, weight: 0.34 },
      { a: a.caution, b: b.caution, weight: 0.16 },
      { a: a.excitability, b: b.excitability, weight: 0.12 },
      { a: a.trainability, b: b.trainability, weight: 0.1 },
    ];
    const available = pairs.filter(
      (pair): pair is { a: number; b: number; weight: number } =>
        pair.a !== null && pair.b !== null,
    );
    if (available.length === 0) {
      return {
        score: this.temperamentSimilarity(rawA, rawB),
        coverage: Math.min(a.coverage, b.coverage),
        socialRisk: this.maxNullable(a.socialRisk, b.socialRisk),
      };
    }

    const totalWeight = available.reduce((sum, pair) => sum + pair.weight, 0);
    let score =
      available.reduce(
        (sum, pair) => sum + (1 - Math.abs(pair.a - pair.b)) * pair.weight,
        0,
      ) / totalWeight;

    const socialRisk = this.maxNullable(a.socialRisk, b.socialRisk);
    if (socialRisk !== null) {
      score -= socialRisk * 0.16;
    }

    return {
      score: this.clamp01(score),
      coverage: Math.min(a.coverage, b.coverage),
      socialRisk,
    };
  }

  private behaviorVector(value: Prisma.JsonValue | null): BehaviorVector {
    const traits = new Map<string, number>();
    if (Array.isArray(value)) {
      for (const raw of value) {
        if (typeof raw === 'string') {
          traits.set(this.normalizeKey(raw), 1);
        }
      }
    } else if (value && this.isJsonObject(value)) {
      for (const [key, raw] of Object.entries(value)) {
        if (typeof raw === 'number') {
          traits.set(this.normalizeKey(key), this.normalizeTraitNumber(raw));
        } else if (typeof raw === 'boolean') {
          traits.set(this.normalizeKey(key), raw ? 1 : 0);
        }
      }
    }

    const trait = (...aliases: string[]) => {
      const present = aliases
        .map((alias) => traits.get(this.normalizeKey(alias)))
        .filter((entry): entry is number => entry !== undefined);
      if (present.length === 0) return null;
      return present.reduce((sum, entry) => sum + entry, 0) / present.length;
    };
    const combine = (...values: Array<number | null>) => {
      const present = values.filter((entry): entry is number => entry !== null);
      if (present.length === 0) return null;
      return present.reduce((sum, entry) => sum + entry, 0) / present.length;
    };

    const energetic = trait('energetic', 'energy', 'energy level', 'active');
    const calm = trait('calm');
    const friendly = trait('friendly', 'social', 'sociable');
    const shy = trait('shy');
    const playful = trait('playful', 'excitable', 'excitability');
    const trainable = trait('trainable', 'trainability');
    const caution = trait(
      'shy',
      'fearful',
      'anxious',
      'dog directed fear',
      'dog-directed fear',
      'protective',
    );
    const socialRisk = trait(
      'aggressive',
      'aggression',
      'dog directed aggression',
      'dog-directed aggression',
    );

    const energy = combine(energetic, calm === null ? null : 1 - calm);
    const sociability = combine(friendly, shy === null ? null : 1 - shy);
    const excitability = combine(playful, energetic);
    const trainability = combine(trainable, calm);
    const dimensions = [energy, sociability, caution, excitability, trainability];

    return {
      energy,
      sociability,
      caution,
      excitability,
      trainability,
      socialRisk,
      coverage:
        dimensions.filter((dimension) => dimension !== null).length /
        dimensions.length,
    };
  }

  private outcomeScore(outcomes: PairOutcome[]): number | null {
    const rated = outcomes.filter((outcome) => outcome.rating !== null);
    const tagged = outcomes.filter((outcome) => outcome.feedbackTags.length > 0);
    if (rated.length === 0 && tagged.length === 0) return null;

    const observations = outcomes.map((outcome) => {
      let score = outcome.rating === null ? 0.65 : (outcome.rating - 1) / 4;
      if (outcome.feedbackTags.includes('great_match')) score += 0.1;
      if (outcome.feedbackTags.includes('owner_friendly')) score += 0.05;
      if (outcome.feedbackTags.includes('energy_mismatch')) score -= 0.12;
      if (outcome.feedbackTags.includes('size_issue')) score -= 0.08;
      if (outcome.feedbackTags.includes('temperament')) score -= 0.18;
      return this.clamp01(score);
    });
    const mean =
      observations.reduce((sum, observation) => sum + observation, 0) /
      observations.length;
    const priorMean = 0.65;
    const priorStrength = 3;
    return (
      (mean * observations.length + priorMean * priorStrength) /
      (observations.length + priorStrength)
    );
  }

  private async getPairOutcomes(ownerAId: string, ownerBId: string) {
    return this.prisma.meetupProposal.findMany({
      where: {
        OR: [
          { proposerId: ownerAId, recipientId: ownerBId },
          { proposerId: ownerBId, recipientId: ownerAId },
        ],
        createdAt: { gte: new Date(Date.now() - 180 * DAY_MS) },
      },
      select: {
        rating: true,
        feedbackTags: true,
        occurredAt: true,
        createdAt: true,
      },
      orderBy: { createdAt: 'desc' },
      take: 20,
    });
  }

  private temperamentSimilarity(
    a: Prisma.JsonValue | null,
    b: Prisma.JsonValue | null,
  ): number | null {
    if (a == null || b == null) return null;

    if (Array.isArray(a) && Array.isArray(b)) {
      const aSet = new Set(
        a
          .filter((item): item is string => typeof item === 'string')
          .map((item) => this.normalizeText(item)),
      );
      const bSet = new Set(
        b
          .filter((item): item is string => typeof item === 'string')
          .map((item) => this.normalizeText(item)),
      );
      if (aSet.size === 0 || bSet.size === 0) return null;
      const intersection = [...aSet].filter((trait) => bSet.has(trait)).length;
      const union = new Set([...aSet, ...bSet]).size;
      return union === 0 ? null : intersection / union;
    }

    if (this.isJsonObject(a) && this.isJsonObject(b)) {
      const sharedKeys = Object.keys(a).filter((key) => key in b);
      if (sharedKeys.length === 0) return null;
      const similarities = sharedKeys.map((key) => {
        const valueA = a[key];
        const valueB = b[key];
        if (typeof valueA === 'number' && typeof valueB === 'number') {
          return 1 - Math.abs(this.normalizeTraitNumber(valueA) - this.normalizeTraitNumber(valueB));
        }
        return this.normalizeText(String(valueA)) === this.normalizeText(String(valueB))
          ? 1
          : 0;
      });
      return similarities.reduce((sum, value) => sum + value, 0) / similarities.length;
    }

    if (typeof a === 'string' && typeof b === 'string') {
      return this.normalizeText(a) === this.normalizeText(b) ? 1 : 0.35;
    }

    return null;
  }

  private ageSimilarity(a: Date | null, b: Date | null): number | null {
    if (!a || !b) return null;
    const years = Math.abs(a.getTime() - b.getTime()) / (365.25 * DAY_MS);
    if (years <= 1) return 1;
    if (years <= 3) return 0.86;
    if (years <= 6) return 0.66;
    return 0.46;
  }

  private temperamentTraits(value: Prisma.JsonValue | null): string[] {
    if (value == null) return [];
    if (Array.isArray(value)) {
      return value
        .filter((item): item is string => typeof item === 'string')
        .slice(0, 6);
    }
    if (this.isJsonObject(value)) {
      return Object.entries(value)
        .filter(([, score]) => typeof score !== 'number' || score >= 3)
        .sort(([, a], [, b]) =>
          (typeof b === 'number' ? b : 0) - (typeof a === 'number' ? a : 0),
        )
        .slice(0, 6)
        .map(([key]) => this.humanizeKey(key));
    }
    return typeof value === 'string' ? [value] : [];
  }

  private async assertActorOwnsEither(
    userId: string,
    petAId: string,
    petBId: string,
  ) {
    const owned = await this.prisma.pet.findFirst({
      where: {
        ownerId: userId,
        id: { in: [petAId, petBId] },
      },
      select: { id: true },
    });
    if (!owned) {
      throw new NotFoundException('Pet relationship not found');
    }
  }

  private isJsonObject(value: Prisma.JsonValue): value is Prisma.JsonObject {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
  }

  private normalizeTraitNumber(value: number) {
    if (value >= 0 && value <= 1) return value;
    if (value >= 1 && value <= 5) return (value - 1) / 4;
    return this.clamp01(value / 4);
  }

  private maxNullable(a: number | null, b: number | null) {
    if (a === null) return b;
    if (b === null) return a;
    return Math.max(a, b);
  }

  private normalizeKey(value: string) {
    return value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  }

  private normalizeText(value: string) {
    return value.trim().toLowerCase();
  }

  private humanizeKey(value: string) {
    return value
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  private clamp01(value: number) {
    return Math.max(0, Math.min(1, value));
  }

  private round(value: number) {
    return Math.round(value * 1000) / 1000;
  }
}
