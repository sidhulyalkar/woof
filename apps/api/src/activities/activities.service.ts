import { BadRequestException, Injectable, Logger, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { CareEventsService } from '../care-events/care-events.service';
import type { WellbeingPathway } from '../care-events/care-event.types';
import { PrismaService } from '../prisma/prisma.service';
import { CreateActivityDto, UpdateActivityDto } from './dto/activity.dto';

@Injectable()
export class ActivitiesService {
  private readonly logger = new Logger(ActivitiesService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly careEvents: CareEventsService
  ) {}

  async create(userId: string, dto: CreateActivityDto) {
    if (dto.petId) {
      await this.assertOwnedPet(userId, dto.petId);
    }

    const startedAt = dto.startedAt ? new Date(dto.startedAt) : new Date();
    const endedAt = dto.endedAt ? new Date(dto.endedAt) : null;
    this.assertChronology(startedAt, endedAt);

    const activity = await this.prisma.activity.create({
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

    if (activity.endedAt && activity.petId) {
      await this.emitActivityCareEvent(userId, activity);
    }

    return activity;
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
    const endedAt = dto.endedAt ? new Date(dto.endedAt) : existing.endedAt;
    this.assertChronology(startedAt, endedAt);

    const activity = await this.prisma.activity.update({
      where: { id },
      data: {
        ...(dto.petId !== undefined ? { petId: dto.petId } : {}),
        ...(dto.startedAt !== undefined ? { startedAt } : {}),
        ...(dto.endedAt !== undefined ? { endedAt } : {}),
        ...(dto.type !== undefined ? { type: dto.type } : {}),
        ...(dto.route !== undefined ? { route: this.json(dto.route) } : {}),
        ...(dto.humanMetrics !== undefined ? { humanMetrics: this.json(dto.humanMetrics) } : {}),
        ...(dto.petMetrics !== undefined ? { petMetrics: this.json(dto.petMetrics) } : {}),
        ...(dto.jointMetrics !== undefined ? { jointMetrics: this.json(dto.jointMetrics) } : {}),
      },
      include: this.activityInclude(),
    });

    if (activity.endedAt && activity.petId) {
      await this.emitActivityCareEvent(userId, activity);
    }

    return activity;
  }

  async delete(userId: string, id: string) {
    await this.assertOwnedActivity(userId, id);
    return this.prisma.activity.delete({ where: { id } });
  }

  private async emitActivityCareEvent(
    userId: string,
    activity: {
      id: string;
      petId: string | null;
      type: string;
      startedAt: Date;
      endedAt: Date | null;
      route: unknown;
    }
  ) {
    if (!activity.petId || !activity.endedAt) return;
    const semantic = this.activitySemantic(activity.type);
    if (!semantic) return;

    const durationMinutes = Math.max(
      0,
      Math.round((activity.endedAt.getTime() - activity.startedAt.getTime()) / 60000)
    );

    try {
      await this.careEvents.record({
        userId,
        petId: activity.petId,
        eventType: semantic.eventType,
        pathway: semantic.pathway,
        occurredAt: activity.endedAt,
        source: 'ACTIVITIES',
        evidenceType: 'ACTIVITY',
        evidenceConfidence: 0.78,
        dedupeKey: `activity:${activity.id}:completed`,
        context: {
          activityId: activity.id,
          activityType: activity.type.toUpperCase(),
          durationMinutes,
          routePresent: Boolean(activity.route),
        },
      });
    } catch (error) {
      // Rewards are additive product infrastructure. Logging an activity must remain
      // reliable during a rolling deployment even if the new ledger migration has
      // not reached this API instance yet.
      this.logger.warn(
        `Activity ${activity.id} saved, but Adventure reward emission failed: ${
          error instanceof Error ? error.message : 'unknown error'
        }`
      );
    }
  }

  private activitySemantic(type: string): {
    eventType: string;
    pathway: WellbeingPathway;
  } | null {
    switch (type.toUpperCase()) {
      case 'WALK':
        return { eventType: 'ACTIVITY_WALK', pathway: 'MOVE' };
      case 'RUN':
        return { eventType: 'ACTIVITY_RUN', pathway: 'MOVE' };
      case 'HIKE':
        return { eventType: 'ACTIVITY_HIKE', pathway: 'EXPLORE' };
      case 'PLAY':
      case 'ENRICHMENT':
      case 'SCENT':
      case 'PUZZLE':
        return { eventType: 'ENRICHMENT_SESSION', pathway: 'ENRICH' };
      case 'TRAINING':
        return { eventType: 'TRAINING_SESSION', pathway: 'LEARN' };
      case 'SOCIAL':
      case 'MEETUP':
        return { eventType: 'SOCIAL_OUTING', pathway: 'CONNECT' };
      case 'PARALLEL_WALK':
        return { eventType: 'PARALLEL_WALK', pathway: 'CONNECT' };
      case 'RECOVERY':
      case 'REST':
      case 'DECOMPRESSION':
        return { eventType: 'RECOVERY_SESSION', pathway: 'RECOVER' };
      default:
        return null;
    }
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
    return value === undefined ? undefined : (value as Prisma.InputJsonValue);
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
