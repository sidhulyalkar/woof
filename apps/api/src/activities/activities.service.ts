import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { Prisma } from '@woof/database';

@Injectable()
export class ActivitiesService {
  constructor(private prisma: PrismaService) {}

  async create(data: Prisma.ActivityCreateInput) {
    return this.prisma.activity.create({
      data,
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
    });
  }

  async findAll(skip = 0, take = 20, userId?: string, petId?: string) {
    const where: Prisma.ActivityWhereInput = {};
    if (userId) where.userId = userId;
    if (petId) where.petId = petId;

    const [activities, total] = await Promise.all([
      this.prisma.activity.findMany({
        where,
        skip,
        take,
        include: {
          user: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
            },
          },
          pet: {
            select: {
              id: true,
              name: true,
              avatarUrl: true,
            },
          },
          _count: {
            select: {
              posts: true,
            },
          },
        },
        orderBy: {
          startedAt: 'desc',
        },
      }),
      this.prisma.activity.count({ where }),
    ]);

    return { activities, total, skip, take };
  }

  async findById(id: string) {
    const activity = await this.prisma.activity.findUnique({
      where: { id },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
            email: true,
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
            species: true,
            avatarUrl: true,
          },
        },
        posts: true,
      },
    });

    if (!activity) {
      throw new NotFoundException(`Activity with ID ${id} not found`);
    }

    return activity;
  }

  async update(id: string, data: Prisma.ActivityUpdateInput) {
    try {
      return await this.prisma.activity.update({
        where: { id },
        data,
        include: {
          user: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
            },
          },
          pet: {
            select: {
              id: true,
              name: true,
              avatarUrl: true,
            },
          },
        },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Activity with ID ${id} not found`);
      }
      throw error;
    }
  }

  async delete(id: string) {
    try {
      return await this.prisma.activity.delete({
        where: { id },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Activity with ID ${id} not found`);
      }
      throw error;
    }
  }
}
