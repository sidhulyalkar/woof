import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';

type CompatibilityPet = {
  id: string;
  name: string;
  species: string;
  breed: string | null;
  birthdate: Date | null;
  temperament: Prisma.JsonValue | null;
};

type CompatibilityFactors = {
  species: number;
  temperament?: number;
  age?: number;
  breed?: number;
};

const BASELINE_VERSION = 'deterministic-baseline-v1';

@Injectable()
export class CompatibilityService {
  constructor(private prisma: PrismaService) {}

  /**
   * Get or create a canonical relationship edge between two different pets.
   */
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

    let edge = await this.prisma.petEdge.findUnique({
      where: {
        petAId_petBId: {
          petAId: firstPetId,
          petBId: secondPetId,
        },
      },
      include: {
        petA: true,
        petB: true,
      },
    });

    if (!edge) {
      edge = await this.prisma.petEdge.create({
        data: {
          petAId: firstPetId,
          petBId: secondPetId,
          weight: 1.0,
          status: 'PROPOSED',
        },
        include: {
          petA: true,
          petB: true,
        },
      });
    }

    return edge;
  }

  /**
   * Calculate a deterministic, explainable compatibility baseline.
   *
   * This baseline is deliberately simple. It gives the product stable behavior
   * during cold start or an ML-service outage, and provides a control that a
   * learned model must beat before promotion.
   */
  async calculateCompatibility(petAId: string, petBId: string) {
    const edge = await this.getOrCreatePetEdge(petAId, petBId);
    const result = this.scoreBaseline(edge.petA, edge.petB);

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

  /**
   * Return ranked recommendations using the stable baseline contract.
   *
   * Existing edges define candidate relationships. Scores are recalculated at
   * read time so legacy placeholder/random scores do not leak into discovery.
   */
  async getRecommendations(petId: string, limit = 10) {
    const safeLimit = Math.max(1, Math.min(Number(limit) || 10, 50));

    const pet = await this.prisma.pet.findUnique({ where: { id: petId } });
    if (!pet) {
      throw new NotFoundException(`Pet with ID ${petId} not found`);
    }

    const edges = await this.prisma.petEdge.findMany({
      where: {
        OR: [{ petAId: petId }, { petBId: petId }],
        status: { not: 'AVOID' },
      },
      include: {
        petA: {
          include: {
            owner: {
              select: {
                id: true,
                handle: true,
                bio: true,
                avatarUrl: true,
                isVerified: true,
              },
            },
          },
        },
        petB: {
          include: {
            owner: {
              select: {
                id: true,
                handle: true,
                bio: true,
                avatarUrl: true,
                isVerified: true,
              },
            },
          },
        },
      },
      // Pull a slightly wider candidate set, then rank by the current baseline.
      take: Math.min(safeLimit * 3, 100),
    });

    const recommendations = edges
      .map((edge) => {
        const matchedPet = edge.petAId === petId ? edge.petB : edge.petA;
        const result = this.scoreBaseline(pet, matchedPet);

        return {
          id: edge.id,
          pet: {
            id: matchedPet.id,
            ownerId: matchedPet.ownerId,
            name: matchedPet.name,
            species: matchedPet.species,
            breed: matchedPet.breed,
            birthdate: matchedPet.birthdate,
            avatarUrl: matchedPet.avatarUrl,
            temperament: this.temperamentTraits(matchedPet.temperament),
            owner: matchedPet.owner,
          },
          compatibilityScore: result.score,
          confidence: result.confidence,
          source: BASELINE_VERSION,
          factors: result.factors,
          explanation: result.explanation,
          status: edge.status,
          lastInteractionAt: edge.lastInteractionAt,
        };
      })
      .sort((a, b) => b.compatibilityScore - a.compatibilityScore)
      .slice(0, safeLimit);

    return {
      petId,
      recommendations,
      total: recommendations.length,
      source: BASELINE_VERSION,
    };
  }

  async updateEdgeStatus(petAId: string, petBId: string, status: string) {
    const validStatuses = ['PROPOSED', 'CONFIRMED', 'AVOID'];
    if (!validStatuses.includes(status)) {
      throw new BadRequestException(
        `Invalid status. Must be one of: ${validStatuses.join(', ')}`,
      );
    }

    const edge = await this.getOrCreatePetEdge(petAId, petBId);

    return this.prisma.petEdge.update({
      where: { id: edge.id },
      data: { status },
      include: {
        petA: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
        petB: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
    });
  }

  async getAllEdges(skip = 0, take = 20, status?: string) {
    const safeSkip = Math.max(0, Number(skip) || 0);
    const safeTake = Math.max(1, Math.min(Number(take) || 20, 100));
    const where: Prisma.PetEdgeWhereInput = status ? { status } : {};

    const [edges, total] = await Promise.all([
      this.prisma.petEdge.findMany({
        where,
        skip: safeSkip,
        take: safeTake,
        include: {
          petA: {
            select: {
              id: true,
              name: true,
              species: true,
              avatarUrl: true,
            },
          },
          petB: {
            select: {
              id: true,
              name: true,
              species: true,
              avatarUrl: true,
            },
          },
        },
        orderBy: {
          lastInteractionAt: 'desc',
        },
      }),
      this.prisma.petEdge.count({ where }),
    ]);

    return { edges, total, skip: safeSkip, take: safeTake };
  }

  private scoreBaseline(petA: CompatibilityPet, petB: CompatibilityPet) {
    const weightedFactors: Array<{ key: keyof CompatibilityFactors; value: number; weight: number }> = [];

    const sameSpecies = this.normalizeText(petA.species) === this.normalizeText(petB.species);
    const speciesScore = sameSpecies ? 1 : 0.15;
    weightedFactors.push({ key: 'species', value: speciesScore, weight: 0.25 });

    const temperamentScore = this.temperamentSimilarity(petA.temperament, petB.temperament);
    if (temperamentScore !== null) {
      weightedFactors.push({ key: 'temperament', value: temperamentScore, weight: 0.5 });
    }

    const ageScore = this.ageSimilarity(petA.birthdate, petB.birthdate);
    if (ageScore !== null) {
      weightedFactors.push({ key: 'age', value: ageScore, weight: 0.15 });
    }

    if (petA.breed && petB.breed) {
      const sameBreed = this.normalizeText(petA.breed) === this.normalizeText(petB.breed);
      // Breed is intentionally a weak signal. Different breeds can be excellent matches.
      weightedFactors.push({ key: 'breed', value: sameBreed ? 0.95 : 0.7, weight: 0.1 });
    }

    const totalWeight = weightedFactors.reduce((sum, factor) => sum + factor.weight, 0);
    const score = weightedFactors.reduce(
      (sum, factor) => sum + factor.value * factor.weight,
      0,
    ) / totalWeight;

    const factors = weightedFactors.reduce<CompatibilityFactors>(
      (acc, factor) => ({ ...acc, [factor.key]: this.round(factor.value) }),
      { species: this.round(speciesScore) },
    );

    const profileCoverage = weightedFactors.reduce((sum, factor) => sum + factor.weight, 0);
    const confidence = Math.min(0.95, 0.45 + profileCoverage * 0.5);

    const explanation: string[] = [];
    if (sameSpecies) {
      explanation.push(`Both pets are ${petA.species.toLowerCase()}s`);
    }
    if (temperamentScore !== null && temperamentScore >= 0.75) {
      explanation.push('Their temperament profiles are strongly aligned');
    } else if (temperamentScore !== null && temperamentScore >= 0.55) {
      explanation.push('Their temperament profiles have useful overlap');
    }
    if (ageScore !== null && ageScore >= 0.85) {
      explanation.push('They are at similar life stages');
    }
    if (petA.breed && petB.breed && this.normalizeText(petA.breed) === this.normalizeText(petB.breed)) {
      explanation.push('They share a breed profile, used only as a light supporting signal');
    }
    if (profileCoverage < 0.75) {
      explanation.push('This estimate is conservative because profile data is still limited');
    }
    if (explanation.length === 0) {
      explanation.push('This is an initial profile-based estimate that will improve with real interaction data');
    }

    return {
      score: this.round(score),
      confidence: this.round(confidence),
      factors,
      explanation,
    };
  }

  private temperamentSimilarity(a: Prisma.JsonValue | null, b: Prisma.JsonValue | null): number | null {
    if (a == null || b == null) return null;

    if (Array.isArray(a) && Array.isArray(b)) {
      const aSet = new Set(a.filter((item): item is string => typeof item === 'string').map(this.normalizeText));
      const bSet = new Set(b.filter((item): item is string => typeof item === 'string').map(this.normalizeText));
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
          // Temperament questionnaires commonly use a 1-5 scale.
          return Math.max(0, 1 - Math.abs(valueA - valueB) / 4);
        }

        return this.normalizeText(String(valueA)) === this.normalizeText(String(valueB)) ? 1 : 0;
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

    const years = Math.abs(a.getTime() - b.getTime()) / (365.25 * 24 * 60 * 60 * 1000);
    if (years <= 1) return 1;
    if (years <= 3) return 0.85;
    if (years <= 6) return 0.65;
    return 0.45;
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
        .sort(([, a], [, b]) => (typeof b === 'number' ? b : 0) - (typeof a === 'number' ? a : 0))
        .slice(0, 6)
        .map(([key]) => this.humanizeKey(key));
    }

    return typeof value === 'string' ? [value] : [];
  }

  private isJsonObject(value: Prisma.JsonValue): value is Prisma.JsonObject {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
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

  private round(value: number) {
    return Math.round(value * 1000) / 1000;
  }
}
