import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { Prisma } from '@woof/database';

@Injectable()
export class PetsService {
  constructor(private prisma: PrismaService) {}

  async create(data: Prisma.PetCreateInput) {
    return this.prisma.pet.create({
      data,
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
    const where: Prisma.PetWhereInput = ownerId ? { ownerId } : {};

    const [pets, total] = await Promise.all([
      this.prisma.pet.findMany({
        where,
        skip,
        take,
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

    return { pets, total, skip, take };
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

  async update(id: string, data: Prisma.PetUpdateInput) {
    try {
      return await this.prisma.pet.update({
        where: { id },
        data,
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
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Pet with ID ${id} not found`);
      }
      throw error;
    }
  }

  async delete(id: string) {
    try {
      return await this.prisma.pet.delete({
        where: { id },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Pet with ID ${id} not found`);
      }
      throw error;
    }
  }
}
