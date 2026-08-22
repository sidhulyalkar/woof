import { BadRequestException, ForbiddenException, Injectable, NotFoundException } from '@nestjs/common';
import { createHash } from 'crypto';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { UpdateHouseholdDto } from './dto/household.dto';

@Injectable()
export class HouseholdsService {
  constructor(private readonly prisma: PrismaService) {}

  async getMine(userId: string) {
    await this.ensurePersonalHousehold(userId);

    const memberships = await this.prisma.householdMember.findMany({
      where: { userId, status: 'ACTIVE' },
      include: {
        household: {
          include: {
            members: {
              where: { status: 'ACTIVE' },
              select: {
                id: true,
                role: true,
                status: true,
                joinedAt: true,
                user: {
                  select: {
                    id: true,
                    handle: true,
                    avatarUrl: true,
                  },
                },
              },
              orderBy: { joinedAt: 'asc' },
            },
            pets: {
              where: { status: 'ACTIVE' },
              select: {
                id: true,
                status: true,
                joinedAt: true,
                pet: {
                  select: {
                    id: true,
                    name: true,
                    species: true,
                    breed: true,
                    birthdate: true,
                    avatarUrl: true,
                  },
                },
              },
              orderBy: { joinedAt: 'asc' },
            },
          },
        },
      },
      orderBy: { joinedAt: 'asc' },
    });

    return memberships.map((membership) => ({
      ...membership.household,
      viewerRole: membership.role,
    }));
  }

  async ensurePersonalHousehold(userId: string): Promise<string> {
    // A dogOS account always owns one deterministic personal household. Do not
    // use the user's first active membership here: once household invitations
    // exist, that could attach a newly created pet to somebody else's household.
    const householdId = this.deterministicUuid(`dogos-household:${userId}`);
    const existing = await this.prisma.householdMember.findUnique({
      where: {
        householdId_userId: {
          householdId,
          userId,
        },
      },
      select: { status: true, role: true },
    });

    if (existing?.status === 'ACTIVE' && existing.role === 'OWNER') {
      return householdId;
    }

    await this.prisma.$transaction(async (tx) => {
      await tx.household.upsert({
        where: { id: householdId },
        update: {},
        create: {
          id: householdId,
          name: 'My household',
        },
      });

      await tx.householdMember.upsert({
        where: {
          householdId_userId: {
            householdId,
            userId,
          },
        },
        update: { status: 'ACTIVE', role: 'OWNER' },
        create: {
          householdId,
          userId,
          role: 'OWNER',
          status: 'ACTIVE',
        },
      });

      const ownedPets = await tx.pet.findMany({
        where: { ownerId: userId },
        select: { id: true },
      });

      for (const pet of ownedPets) {
        await tx.householdPet.upsert({
          where: {
            householdId_petId: {
              householdId,
              petId: pet.id,
            },
          },
          update: { status: 'ACTIVE' },
          create: {
            householdId,
            petId: pet.id,
            status: 'ACTIVE',
          },
        });
      }
    });

    return householdId;
  }

  async update(userId: string, householdId: string, dto: UpdateHouseholdDto) {
    await this.assertCanManage(userId, householdId);

    return this.prisma.household.update({
      where: { id: householdId },
      data: {
        ...(dto.name !== undefined ? { name: dto.name.trim() } : {}),
        ...(dto.timezone !== undefined ? { timezone: dto.timezone.trim() } : {}),
      },
    });
  }

  async addOwnedPet(userId: string, householdId: string, petId: string) {
    await this.assertCanManage(userId, householdId);

    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true },
    });
    if (!pet) throw new NotFoundException('Pet not found');

    return this.prisma.householdPet.upsert({
      where: {
        householdId_petId: {
          householdId,
          petId,
        },
      },
      update: { status: 'ACTIVE' },
      create: {
        householdId,
        petId,
        status: 'ACTIVE',
      },
      include: {
        pet: {
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
  }

  async removePet(userId: string, householdId: string, petId: string) {
    await this.assertCanManage(userId, householdId);

    const link = await this.prisma.householdPet.findUnique({
      where: {
        householdId_petId: {
          householdId,
          petId,
        },
      },
      select: { id: true },
    });
    if (!link) throw new NotFoundException('Household pet not found');

    return this.prisma.householdPet.update({
      where: { id: link.id },
      data: { status: 'INACTIVE' },
    });
  }

  async attachNewOwnedPet(userId: string, petId: string) {
    const householdId = await this.ensurePersonalHousehold(userId);
    await this.prisma.householdPet.upsert({
      where: {
        householdId_petId: {
          householdId,
          petId,
        },
      },
      update: { status: 'ACTIVE' },
      create: {
        householdId,
        petId,
        status: 'ACTIVE',
      },
    });
    return householdId;
  }

  async assertPetAccessible(userId: string, petId: string) {
    const pet = await this.prisma.pet.findFirst({
      where: {
        id: petId,
        OR: [
          { ownerId: userId },
          {
            householdMemberships: {
              some: {
                status: 'ACTIVE',
                household: {
                  members: {
                    some: {
                      userId,
                      status: 'ACTIVE',
                    },
                  },
                },
              },
            },
          },
        ],
      },
      select: { id: true },
    });

    if (!pet) throw new NotFoundException('Pet not found');
    return pet;
  }

  async resolveActivityHousehold(
    userId: string,
    petIds: string[],
    requestedHouseholdId?: string
  ): Promise<string> {
    const uniquePetIds = [...new Set(petIds)];

    const memberships = await this.prisma.householdMember.findMany({
      where: {
        userId,
        status: 'ACTIVE',
        ...(requestedHouseholdId ? { householdId: requestedHouseholdId } : {}),
      },
      select: {
        householdId: true,
        household: {
          select: {
            pets: {
              where: { status: 'ACTIVE' },
              select: { petId: true },
            },
          },
        },
      },
      orderBy: { joinedAt: 'asc' },
    });

    if (!memberships.length && requestedHouseholdId) {
      throw new NotFoundException('Household not found');
    }

    if (!memberships.length) {
      await this.ensurePersonalHousehold(userId);
      return this.resolveActivityHousehold(userId, uniquePetIds, requestedHouseholdId);
    }

    if (!uniquePetIds.length) return memberships[0].householdId;

    for (const membership of memberships) {
      const accessible = new Set(membership.household.pets.map((link) => link.petId));
      if (uniquePetIds.every((petId) => accessible.has(petId))) {
        return membership.householdId;
      }
    }

    for (const petId of uniquePetIds) {
      await this.assertPetAccessible(userId, petId);
    }

    throw new BadRequestException('Selected pets must belong to one shared household');
  }

  householdActivityWhere(userId: string): Prisma.ActivityWhereInput {
    return {
      OR: [
        { userId },
        {
          household: {
            members: {
              some: {
                userId,
                status: 'ACTIVE',
              },
            },
          },
        },
      ],
    };
  }

  private async assertCanManage(userId: string, householdId: string) {
    const membership = await this.prisma.householdMember.findFirst({
      where: {
        householdId,
        userId,
        status: 'ACTIVE',
      },
      select: { role: true },
    });

    if (!membership) throw new NotFoundException('Household not found');
    if (!['OWNER', 'ADMIN'].includes(membership.role)) {
      throw new ForbiddenException('Household manager access required');
    }
  }

  private deterministicUuid(input: string) {
    const hex = createHash('md5').update(input).digest('hex');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
}
