import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { CreatePetDto, UpdatePetDto } from './dto/create-pet.dto';

@Injectable()
export class PetsService {
  constructor(private prisma: PrismaService) {}

  async create(ownerId: string, data: CreatePetDto) {
    return this.prisma.pet.create({
      data: this.toCreateInput(ownerId, data),
      include: {
        owner: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
      },
    });
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
      include: {
        owner: {
          select: {
            id: true,
            handle: true,
            email: true,
            avatarUrl: true,
          },
        },
        devices: true,
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
