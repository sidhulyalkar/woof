import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { Prisma } from '@woof/database';

@Injectable()
export class CompatibilityService {
  constructor(private prisma: PrismaService) {}

  /**
   * Get or create a pet edge (relationship between two pets)
   */
  async getOrCreatePetEdge(petAId: string, petBId: string) {
    // Ensure petAId < petBId for consistency
    const [firstPetId, secondPetId] = [petAId, petBId].sort();

    // Check if pets exist
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

    // Get or create the edge
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
   * Calculate compatibility score between two pets
   * This is a placeholder for ML model integration
   */
  async calculateCompatibility(petAId: string, petBId: string) {
    const edge = await this.getOrCreatePetEdge(petAId, petBId);

    // TODO: Replace with actual ML model prediction
    // For now, return a simple mock score based on temperament similarity
    const petA = edge.petA;
    const petB = edge.petB;

    let score = 0.5; // Base score

    // Check species match
    if (petA.species === petB.species) {
      score += 0.2;
    }

    // Check temperament (if available)
    if (petA.temperament && petB.temperament) {
      // Placeholder: just add a random factor
      score += Math.random() * 0.3;
    }

    // Normalize score to 0-1 range
    score = Math.max(0, Math.min(1, score));

    // Update the edge with the calculated score
    const updatedEdge = await this.prisma.petEdge.update({
      where: { id: edge.id },
      data: {
        compatibilityScore: score,
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
      compatibilityScore: score,
      edge: updatedEdge,
      message: 'Compatibility score calculated (using placeholder algorithm)',
    };
  }

  /**
   * Get compatibility recommendations for a pet
   */
  async getRecommendations(petId: string, limit = 10) {
    // Check if pet exists
    const pet = await this.prisma.pet.findUnique({ where: { id: petId } });
    if (!pet) {
      throw new NotFoundException(`Pet with ID ${petId} not found`);
    }

    // Get all edges for this pet
    const edges = await this.prisma.petEdge.findMany({
      where: {
        OR: [{ petAId: petId }, { petBId: petId }],
        compatibilityScore: { not: null },
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
      orderBy: {
        compatibilityScore: 'desc',
      },
      take: limit,
    });

    // Format the recommendations
    const recommendations = edges.map((edge) => {
      const isSourcePetA = edge.petAId === petId;
      const matchedPet = isSourcePetA ? edge.petB : edge.petA;

      return {
        pet: matchedPet,
        compatibilityScore: edge.compatibilityScore,
        status: edge.status,
        lastInteractionAt: edge.lastInteractionAt,
      };
    });

    return {
      petId,
      recommendations,
      total: recommendations.length,
    };
  }

  /**
   * Update pet edge status (PROPOSED, CONFIRMED, AVOID)
   */
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

  /**
   * Get all pet edges (for analytics/debugging)
   */
  async getAllEdges(skip = 0, take = 20, status?: string) {
    const where: Prisma.PetEdgeWhereInput = status ? { status } : {};

    const [edges, total] = await Promise.all([
      this.prisma.petEdge.findMany({
        where,
        skip,
        take,
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
          compatibilityScore: 'desc',
        },
      }),
      this.prisma.petEdge.count({ where }),
    ]);

    return { edges, total, skip, take };
  }
}
