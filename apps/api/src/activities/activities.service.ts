import {
  BadRequestException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { CreateActivityDto, UpdateActivityDto } from './dto/activity.dto';

@Injectable()
export class ActivitiesService {
  constructor(private readonly prisma: PrismaService) {}

  async create(userId: string, dto: CreateActivityDto) {
    if (dto.petId) {
      await this.assertOwnedPet(userId, dto.petId);
    }

    const startedAt = dto.startedAt ? new Date(dto.startedAt) : new Date();
    const endedAt = dto.endedAt ? new Date(dto.endedAt) : null;
    this.assertChronology(startedAt, endedAt);

    return this.prisma.activity.create({
      data: {
        userId,
        petId: dto.petId,
        startedAt,
        endedAt,
        type: dto.type,
        route: this.json(dto.route),
        humanMetrics: this.json(dto.humanMetrics),
        petMetrics: this.json(dto.petMetrics),
        jointMetrics: this.json(dto.jointMetrics),
      },
      include: this.activityInclude(),
    });
  }

  async findAll(userId: string, skip = 0, take = 20, petId?: string) {
    const safeSkip = Math.max(0, Number(skip) || 0);
    const safeTake = Math.max(1, Math.min(Number(take) || 20, 100));

    if (petId) {
      await this.assertOwnedPet(userId, petId);
    }

    const where: Prisma.ActivityWhereInput = {
      userId,
      ...(petId ? { petId } : {}),
    };

    const [activities, total] = await Promise.all([
      this.prisma.activity.findMany({
        where,
        skip: safeSkip,
        take: safeTake,
        include: {
          ...this.activityInclude(),
          _count: { select: { posts: true } },
        },
        orderBy: { startedAt: 'desc' },
      }),
      this.prisma.activity.count({ where }),
    ]);

    return { activities, total, skip: safeSkip, take: safeTake };
  }

  async findById(userId: string, id: string) {
    const activity = await this.prisma.activity.findFirst({
      where: { id, userId },
      include: {
        ...this.activityInclude(),
        posts: {
          select: {
            id: true,
            text: true,
            mediaUrls: true,
            visibility: true,
            createdAt: true,
          },
        },
      },
    });

    if (!activity) {
      throw new NotFoundException(`Activity with ID ${id} not found`);
    }

    return activity;
  }

  async update(userId: string, id: string, dto: UpdateActivityDto) {
    const existing = await this.assertOwnedActivity(userId, id);

    if (dto.petId) {
      await this.assertOwnedPet(userId, dto.petId);
    }

    const startedAt = dto.startedAt ? new Date(dto.startedAt) : existing.startedAt;
    const endedAt = dto.endedAt
      ? new Date(dto.endedAt)
      : existing.endedAt;
    this.assertChronology(startedAt, endedAt);

    return this.prisma.activity.update({
      where: { id },
      data: {
        ...(dto.petId !== undefined ? { petId: dto.petId } : {}),
        ...(dto.startedAt !== undefined ? { startedAt } : {}),
        ...(dto.endedAt !== undefined ? { endedAt } : {}),
        ...(dto.type !== undefined ? { type: dto.type } : {}),
        ...(dto.route !== undefined ? { route: this.json(dto.route) } : {}),
        ...(dto.humanMetrics !== undefined
          ? { humanMetrics: this.json(dto.humanMetrics) }
          : {}),
        ...(dto.petMetrics !== undefined
          ? { petMetrics: this.json(dto.petMetrics) }
          : {}),
        ...(dto.jointMetrics !== undefined
          ? { jointMetrics: this.json(dto.jointMetrics) }
          : {}),
      },
      include: this.activityInclude(),
    });
  }

  async delete(userId: string, id: string) {
    await this.assertOwnedActivity(userId, id);
    return this.prisma.activity.delete({ where: { id } });
  }

  private async assertOwnedPet(userId: string, petId: string) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true },
    });
    if (!pet) {
      throw new NotFoundException('Pet not found');
    }
    return pet;
  }

  private async assertOwnedActivity(userId: string, id: string) {
    const activity = await this.prisma.activity.findFirst({
      where: { id, userId },
      select: { id: true, startedAt: true, endedAt: true },
    });
    if (!activity) {
      throw new NotFoundException(`Activity with ID ${id} not found`);
    }
    return activity;
  }

  private assertChronology(startedAt: Date, endedAt: Date | null) {
    if (endedAt && endedAt.getTime() < startedAt.getTime()) {
      throw new BadRequestException('Activity end time cannot precede start time');
    }
  }

  private json(value?: Record<string, unknown>) {
    return value === undefined
      ? undefined
      : (value as Prisma.InputJsonValue);
  }

  private activityInclude() {
    return {
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
          species: true,
          avatarUrl: true,
        },
      },
    } satisfies Prisma.ActivityInclude;
  }
}
