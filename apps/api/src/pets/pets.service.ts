import {
  BadRequestException,
  ConflictException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { createHash } from 'crypto';
import { Prisma } from '@woof/database';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { CreatePetDto, UpdatePetDto } from './dto/create-pet.dto';

@Injectable()
export class PetsService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly households: HouseholdsService
  ) {}

  async create(ownerId: string, data: CreatePetDto) {
    const householdId = await this.households.ensurePersonalHousehold(ownerId);
    const replaySafeId = data.creationKey ? this.replaySafePetId(ownerId, data.creationKey) : null;

    if (data.creationKey && (data.temperament || data.vaccinations || data.avatarUrl)) {
      throw new BadRequestException(
        'Replay-safe pet creation accepts durable identity fields only; attach media and mutable profile data after creation'
      );
    }

    if (replaySafeId) {
      const existing = await this.readReplayPet(replaySafeId);
      if (existing) {
        this.assertCreationReplayMatches(existing, ownerId, data);
        return existing;
      }
    }

    const create = () =>
      this.prisma.pet.create({
        data: {
          ...(replaySafeId ? { id: replaySafeId } : {}),
          ...this.toCreateInput(ownerId, data),
          householdMemberships: {
            create: {
              householdId,
              status: 'ACTIVE',
            },
          },
        },
        include: {
          owner: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
              isVerified: true,
            },
          },
          householdMemberships: {
            where: { status: 'ACTIVE' },
            select: { householdId: true },
          },
        },
      });

    try {
      return await create();
    } catch (error) {
      if (!replaySafeId || !this.isUniqueViolation(error)) throw error;

      const existing = await this.readReplayPet(replaySafeId);
      if (!existing) throw error;
      this.assertCreationReplayMatches(existing, ownerId, data);
      return existing;
    }
  }

  async findAll(skip = 0, take = 20, ownerId?: string) {
    const safeSkip = Math.max(0, Number(skip) || 0);
    const safeTake = Math.max(1, Math.min(Number(take) || 20, 100));
    const where: Prisma.PetWhereInput = ownerId ? { ownerId } : {};

    const [pets, total] = await Promise.all([
      this.prisma.pet.findMany({
        where,
        skip: safeSkip,
        take: safeTake,
        include: {
          owner: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
              isVerified: true,
            },
          },
          _count: {
            select: {
              activities: true,
              posts: true,
            },
          },
        },
        orderBy: {
          createdAt: 'desc',
        },
      }),
      this.prisma.pet.count({ where }),
    ]);

    return { pets, total, skip: safeSkip, take: safeTake };
  }

  async findById(id: string) {
    const pet = await this.prisma.pet.findUnique({
      where: { id },
      select: {
        id: true,
        ownerId: true,
        name: true,
        species: true,
        breed: true,
        sex: true,
        birthdate: true,
        temperament: true,
        vaccinations: true,
        avatarUrl: true,
        createdAt: true,
        updatedAt: true,
        owner: {
          select: {
            id: true,
            handle: true,
            bio: true,
            avatarUrl: true,
            isVerified: true,
          },
        },
        _count: {
          select: {
            activities: true,
            posts: true,
            mutualGoals: true,
          },
        },
      },
    });

    if (!pet) {
      throw new NotFoundException(`Pet with ID ${id} not found`);
    }

    return pet;
  }

  async updateOwned(id: string, ownerId: string, data: UpdatePetDto) {
    await this.assertOwned(id, ownerId);

    return this.prisma.pet.update({
      where: { id },
      data: this.toUpdateInput(data),
      include: {
        owner: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
            isVerified: true,
          },
        },
      },
    });
  }

  async deleteOwned(id: string, ownerId: string) {
    await this.assertOwned(id, ownerId);
    return this.prisma.pet.delete({ where: { id } });
  }

  private async assertOwned(id: string, ownerId: string) {
    const pet = await this.prisma.pet.findFirst({
      where: { id, ownerId },
      select: { id: true },
    });

    // Do not reveal whether another user's pet exists through a mutation endpoint.
    if (!pet) {
      throw new NotFoundException(`Pet with ID ${id} not found`);
    }
  }

  private async readReplayPet(id: string) {
    return this.prisma.pet.findUnique({
      where: { id },
      include: {
        owner: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
            isVerified: true,
          },
        },
        householdMemberships: {
          where: { status: 'ACTIVE' },
          select: { householdId: true },
        },
      },
    });
  }

  private assertCreationReplayMatches(
    existing: {
      ownerId: string;
      name: string;
      species: string;
      breed: string | null;
      sex: string | null;
      birthdate: Date | null;
    },
    ownerId: string,
    data: CreatePetDto
  ) {
    const expectedBirthdate = data.birthdate ? new Date(data.birthdate).getTime() : null;
    const existingBirthdate = existing.birthdate?.getTime() ?? null;
    const matches =
      existing.ownerId === ownerId &&
      existing.name === data.name.trim() &&
      existing.species === data.species.trim().toUpperCase() &&
      (existing.breed ?? null) === (data.breed?.trim() || null) &&
      (existing.sex ?? null) === (data.sex ?? null) &&
      existingBirthdate === expectedBirthdate;

    if (!matches) {
      throw new ConflictException('Pet creation key was replayed with divergent identity fields');
    }
  }

  private replaySafePetId(ownerId: string, creationKey: string): string {
    const digest = createHash('sha256')
      .update(`woof-pet-create-v1:${ownerId}:${creationKey.trim()}`)
      .digest('hex');
    return `pet_${digest.slice(0, 32)}`;
  }

  private isUniqueViolation(error: unknown): boolean {
    return error instanceof Prisma.PrismaClientKnownRequestError && error.code === 'P2002';
  }

  private toCreateInput(ownerId: string, data: CreatePetDto): Prisma.PetCreateInput {
    return {
      owner: { connect: { id: ownerId } },
      name: data.name.trim(),
      species: data.species.trim().toUpperCase(),
      breed: data.breed?.trim() || undefined,
      sex: data.sex,
      birthdate: data.birthdate ? new Date(data.birthdate) : undefined,
      temperament: data.temperament as Prisma.InputJsonValue | undefined,
      vaccinations: data.vaccinations as Prisma.InputJsonValue | undefined,
      avatarUrl: data.avatarUrl,
    };
  }

  private toUpdateInput(data: UpdatePetDto): Prisma.PetUpdateInput {
    return {
      ...(data.name !== undefined ? { name: data.name.trim() } : {}),
      ...(data.species !== undefined ? { species: data.species.trim().toUpperCase() } : {}),
      ...(data.breed !== undefined ? { breed: data.breed.trim() || null } : {}),
      ...(data.sex !== undefined ? { sex: data.sex } : {}),
      ...(data.birthdate !== undefined ? { birthdate: new Date(data.birthdate) } : {}),
      ...(data.temperament !== undefined
        ? { temperament: data.temperament as Prisma.InputJsonValue }
        : {}),
      ...(data.vaccinations !== undefined
        ? { vaccinations: data.vaccinations as Prisma.InputJsonValue }
        : {}),
      ...(data.avatarUrl !== undefined ? { avatarUrl: data.avatarUrl } : {}),
    };
  }
}
