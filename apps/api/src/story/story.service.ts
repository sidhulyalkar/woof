import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import type { StoryQueryDto, UpdateStoryCurationDto } from './dto/story.dto';
import type {
  StoryCurationPayload,
  StoryLifeStats,
  StoryMilestone,
  StoryMoment,
  StorySourceType,
} from './story.types';

const STORY_SCAN_PER_SOURCE = 150;
const STORY_STATS_ACTIVITY_SCAN = 5000;

type CurationRow = {
  id: string;
  payload: Prisma.JsonValue;
};

function jsonObject(value: Prisma.JsonValue | null | undefined): Prisma.JsonObject | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return value as Prisma.JsonObject;
}

function readString(value: Prisma.JsonValue | undefined): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function readNumber(value: Prisma.JsonValue | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function readBoolean(value: Prisma.JsonValue | undefined): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

function curationKey(sourceType: StorySourceType, sourceId: string) {
  return `${sourceType}:${sourceId}`;
}

function readCuration(value: Prisma.JsonValue): StoryCurationPayload | null {
  const object = jsonObject(value);
  if (!object || object.schemaVersion !== 'dogos-story-curation-v1') return null;
  const sourceType = readString(object.sourceType);
  const sourceId = readString(object.sourceId);
  const state = readString(object.state);
  const updatedAt = readString(object.updatedAt);
  if (
    (sourceType !== 'ACTIVITY' && sourceType !== 'CARE_EVENT' && sourceType !== 'MEDIA') ||
    !sourceId ||
    (state !== 'SAVED' && state !== 'HIDDEN') ||
    !updatedAt
  ) {
    return null;
  }
  const note = readString(object.note);
  return {
    schemaVersion: 'dogos-story-curation-v1',
    sourceType,
    sourceId,
    state,
    ...(note ? { note } : {}),
    updatedAt,
  };
}

function inputJson(value: StoryCurationPayload): Prisma.InputJsonObject {
  return value as unknown as Prisma.InputJsonObject;
}

function humanize(value: string) {
  return value
    .toLowerCase()
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

@Injectable()
export class StoryService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly households: HouseholdsService
  ) {}

  async getStory(userId: string, query: StoryQueryDto) {
    const petId = query.petId;
    if (petId) await this.households.assertPetAccessible(userId, petId);

    const limit = Math.max(1, Math.min(query.limit ?? 50, 100));
    const before = query.before ? new Date(query.before) : null;
    const accessiblePetIds = await this.accessiblePetIds(userId);
    const activityWhere = this.activityWhere(userId, petId, before);
    const careWhere = this.careWhere(userId, accessiblePetIds, petId, before);
    const mediaWhere = this.mediaWhere(userId, petId, before);

    const [activities, careEvents, mediaAssets, curationRows, stats] = await Promise.all([
      this.prisma.activity.findMany({
        where: activityWhere,
        take: STORY_SCAN_PER_SOURCE,
        orderBy: { startedAt: 'desc' },
        select: {
          id: true,
          petId: true,
          type: true,
          startedAt: true,
          endedAt: true,
          route: true,
          pet: { select: { id: true, name: true } },
          petParticipants: {
            select: { petId: true, pet: { select: { name: true } } },
          },
        },
      }),
      this.prisma.careEvent.findMany({
        where: careWhere,
        take: STORY_SCAN_PER_SOURCE,
        orderBy: { occurredAt: 'desc' },
        select: {
          id: true,
          petId: true,
          eventType: true,
          pathway: true,
          occurredAt: true,
          source: true,
          context: true,
          outcome: true,
          pet: { select: { name: true } },
        },
      }),
      this.prisma.mediaAsset.findMany({
        where: mediaWhere,
        take: STORY_SCAN_PER_SOURCE,
        orderBy: [{ capturedAt: 'desc' }, { createdAt: 'desc' }],
        select: {
          id: true,
          petId: true,
          filename: true,
          mediaType: true,
          capturedAt: true,
          createdAt: true,
          favorite: true,
          pet: { select: { name: true } },
        },
      }),
      this.prisma.notification.findMany({
        where: { userId, type: 'STORY_CURATION' },
        take: 2000,
        orderBy: { createdAt: 'desc' },
        select: { id: true, payload: true },
      }),
      this.getLifeStats(userId, petId),
    ]);

    const curations = new Map<string, StoryCurationPayload>();
    for (const row of curationRows) {
      const curation = readCuration(row.payload);
      if (!curation) continue;
      const key = curationKey(curation.sourceType, curation.sourceId);
      if (!curations.has(key)) curations.set(key, curation);
    }

    const activityMoments = activities.map((activity) => {
      const petPairs = new Map<string, string>();
      if (activity.petId && activity.pet) petPairs.set(activity.petId, activity.pet.name);
      for (const participant of activity.petParticipants) {
        petPairs.set(participant.petId, participant.pet.name);
      }
      const duration = activity.endedAt
        ? Math.max(0, Math.round((activity.endedAt.getTime() - activity.startedAt.getTime()) / 60000))
        : null;
      const label = humanize(activity.type);
      return this.decorateMoment(
        {
          id: `activity:${activity.id}`,
          sourceType: 'ACTIVITY',
          sourceId: activity.id,
          petIds: [...petPairs.keys()],
          petNames: [...petPairs.values()],
          occurredAt: activity.startedAt.toISOString(),
          kind: activity.type,
          title: label,
          summary: duration ? `${duration} minutes together.` : `${label} recorded together.`,
          suggested: activity.type === 'HIKE' || activity.type === 'MEETUP',
          curation: { state: null, note: null },
        },
        curations
      );
    });

    const careMoments = careEvents
      .filter((event) => {
        const context = jsonObject(event.context);
        // Activities already appear from the canonical Activity record. Their
        // reward-side CareEvent is evidence, not a second life-story moment.
        if (readString(context?.activityId)) return false;
        // Device upkeep is useful to Autopilot, but it is not part of the dog's
        // life narrative. Daily wearable summaries remain eligible as context.
        return event.eventType !== 'TRACKER_DEVICE_STATUS';
      })
      .map((event) => {
        const context = jsonObject(event.context);
        const outcome = jsonObject(event.outcome);
        const trackerMinutes = readNumber(context?.activityMinutes);
        const dogExperience = readString(outcome?.dogExperience);
        const title = this.careTitle(event.eventType, event.pathway);
        const summary =
          event.eventType === 'TRACKER_DAILY_ACTIVITY' && trackerMinutes !== undefined
            ? `Tracker summary: ${Math.round(trackerMinutes)} active minutes. This is context, not a health judgment.`
            : dogExperience
              ? `Reflection saved: ${humanize(dogExperience)}.`
              : `${humanize(event.pathway)} experience recorded.`;
        return this.decorateMoment(
          {
            id: `care:${event.id}`,
            sourceType: 'CARE_EVENT',
            sourceId: event.id,
            petIds: event.petId ? [event.petId] : [],
            petNames: event.pet?.name ? [event.pet.name] : [],
            occurredAt: event.occurredAt.toISOString(),
            kind: event.eventType,
            title,
            summary,
            pathway: event.pathway,
            suggested:
              readBoolean(context?.newPlace) === true ||
              readBoolean(context?.memoryAdded) === true ||
              event.eventType === 'SAFE_OPT_OUT',
            curation: { state: null, note: null },
          },
          curations
        );
      });

    const mediaMoments = mediaAssets.map((asset) =>
      this.decorateMoment(
        {
          id: `media:${asset.id}`,
          sourceType: 'MEDIA',
          sourceId: asset.id,
          petIds: [asset.petId],
          petNames: [asset.pet.name],
          occurredAt: (asset.capturedAt ?? asset.createdAt).toISOString(),
          kind: asset.mediaType.toUpperCase(),
          title: asset.mediaType === 'video' ? 'Video memory' : 'Photo memory',
          summary: asset.filename,
          mediaType: asset.mediaType,
          favorite: asset.favorite,
          suggested: asset.favorite,
          curation: { state: null, note: null },
        },
        curations
      )
    );

    const combined = [...activityMoments, ...careMoments, ...mediaMoments]
      .filter((moment) => moment.curation.state !== 'HIDDEN')
      .sort((a, b) => new Date(b.occurredAt).getTime() - new Date(a.occurredAt).getTime());
    const moments = combined.slice(0, limit);
    const nextBefore = combined.length > limit ? moments.at(-1)?.occurredAt ?? null : null;

    return {
      moments,
      milestones: stats.milestones,
      stats: stats.stats,
      nextBefore,
      principles: [
        'Source records stay authoritative; Story stores only presentation and curation state.',
        'Private care context stays private unless its source explicitly permits household visibility.',
        'Wearable summaries are context, never diagnoses or Bond XP.',
        'Suggestions are optional. Saving, hiding, and notes remain user-controlled.',
      ],
    };
  }

  async updateCuration(userId: string, dto: UpdateStoryCurationDto) {
    await this.assertSourceAccessible(userId, dto.sourceType, dto.sourceId);
    const lockKey = `dogos-story-curation:${userId}:${dto.sourceType}:${dto.sourceId}`;

    return this.prisma.$transaction(async (tx) => {
      await tx.$queryRaw<Array<{ acquired: number }>>(Prisma.sql`
        WITH lock_row AS MATERIALIZED (
          SELECT pg_advisory_xact_lock(hashtextextended(${lockKey}, 0))
        )
        SELECT 1::int AS acquired FROM lock_row
      `);

      const existing = await tx.$queryRaw<CurationRow[]>(Prisma.sql`
        SELECT id, payload
        FROM notifications
        WHERE user_id = ${userId}
          AND type = 'STORY_CURATION'
          AND payload->>'sourceType' = ${dto.sourceType}
          AND payload->>'sourceId' = ${dto.sourceId}
        LIMIT 1
      `);

      if (dto.action === 'CLEAR') {
        if (existing[0]) await tx.notification.delete({ where: { id: existing[0].id } });
        return { sourceType: dto.sourceType, sourceId: dto.sourceId, state: null, note: null };
      }

      const payload: StoryCurationPayload = {
        schemaVersion: 'dogos-story-curation-v1',
        sourceType: dto.sourceType,
        sourceId: dto.sourceId,
        state: dto.action === 'SAVE' ? 'SAVED' : 'HIDDEN',
        ...(dto.note?.trim() ? { note: dto.note.trim() } : {}),
        updatedAt: new Date().toISOString(),
      };

      if (existing[0]) {
        await tx.notification.update({
          where: { id: existing[0].id },
          data: { payload: inputJson(payload) },
        });
      } else {
        await tx.notification.create({
          data: { userId, type: 'STORY_CURATION', payload: inputJson(payload) },
        });
      }

      return {
        sourceType: payload.sourceType,
        sourceId: payload.sourceId,
        state: payload.state,
        note: payload.note ?? null,
      };
    });
  }

  private decorateMoment(
    moment: StoryMoment,
    curations: Map<string, StoryCurationPayload>
  ): StoryMoment {
    const curation = curations.get(curationKey(moment.sourceType, moment.sourceId));
    return {
      ...moment,
      curation: {
        state: curation?.state ?? null,
        note: curation?.note ?? null,
      },
    };
  }

  private async getLifeStats(userId: string, petId?: string) {
    const activityWhere = this.activityWhere(userId, petId, null);
    const mediaWhere = this.mediaWhere(userId, petId, null);
    const [activityCount, activities, memoryCount] = await Promise.all([
      this.prisma.activity.count({ where: activityWhere }),
      this.prisma.activity.findMany({
        where: activityWhere,
        take: STORY_STATS_ACTIVITY_SCAN,
        orderBy: { startedAt: 'asc' },
        select: {
          id: true,
          startedAt: true,
          endedAt: true,
          route: true,
          humanMetrics: true,
          petMetrics: true,
          jointMetrics: true,
        },
      }),
      this.prisma.mediaAsset.count({ where: mediaWhere }),
    ]);

    let activeMinutes = 0;
    let distanceMeters = 0;
    const namedPlaces = new Set<string>();
    for (const activity of activities) {
      if (activity.endedAt) {
        activeMinutes += Math.max(
          0,
          (activity.endedAt.getTime() - activity.startedAt.getTime()) / 60000
        );
      }
      distanceMeters += this.activityDistance(activity);
      const place = this.namedPlace(activity.route);
      if (place) namedPlaces.add(place.toLowerCase());
    }

    const stats: StoryLifeStats = {
      activities: activityCount,
      activeMinutes: Math.round(activeMinutes),
      distanceMeters: Math.round(distanceMeters),
      memories: memoryCount,
      namedPlaces: namedPlaces.size,
      coverage: activityCount > STORY_STATS_ACTIVITY_SCAN ? 'BOUNDED' : 'COMPLETE',
    };

    return { stats, milestones: this.milestones(activities, memoryCount) };
  }

  private milestones(
    activities: Array<{ startedAt: Date; endedAt: Date | null }>,
    memoryCount: number
  ): StoryMilestone[] {
    const milestones: StoryMilestone[] = [];
    const first = activities[0];
    if (first) {
      milestones.push({
        id: 'first-adventure',
        title: 'First recorded adventure',
        description: 'The beginning of this Woof life record.',
        achievedAt: first.startedAt.toISOString(),
      });
    }

    for (const threshold of [10, 50, 100]) {
      const activity = activities[threshold - 1];
      if (!activity) continue;
      milestones.push({
        id: `${threshold}-activities`,
        title: `${threshold} shared activities`,
        description: `A real-world rhythm built across ${threshold} recorded activities.`,
        achievedAt: activity.startedAt.toISOString(),
      });
    }

    let runningMinutes = 0;
    const hourThresholds = [10, 50, 100];
    let hourIndex = 0;
    for (const activity of activities) {
      if (activity.endedAt) {
        runningMinutes += Math.max(
          0,
          (activity.endedAt.getTime() - activity.startedAt.getTime()) / 60000
        );
      }
      while (hourIndex < hourThresholds.length && runningMinutes >= hourThresholds[hourIndex] * 60) {
        const hours = hourThresholds[hourIndex];
        milestones.push({
          id: `${hours}-hours`,
          title: `${hours} hours together`,
          description: `${hours} recorded hours of shared activity.`,
          achievedAt: activity.startedAt.toISOString(),
        });
        hourIndex += 1;
      }
    }

    if (memoryCount >= 1) {
      milestones.push({
        id: 'first-memory',
        title: 'First kept memory',
        description: 'At least one private media memory now lives in the library.',
        achievedAt: first?.startedAt.toISOString() ?? new Date(0).toISOString(),
      });
    }

    return milestones.sort(
      (a, b) => new Date(b.achievedAt).getTime() - new Date(a.achievedAt).getTime()
    );
  }

  private activityDistance(activity: {
    humanMetrics: Prisma.JsonValue;
    petMetrics: Prisma.JsonValue;
    jointMetrics: Prisma.JsonValue;
  }) {
    const metricObjects = [
      jsonObject(activity.humanMetrics),
      jsonObject(activity.petMetrics),
      jsonObject(activity.jointMetrics),
    ];
    for (const metrics of metricObjects) {
      const meters = readNumber(metrics?.distanceMeters) ?? readNumber(metrics?.distance_meters);
      if (meters !== undefined && meters >= 0) return meters;
      const kilometers = readNumber(metrics?.distanceKm) ?? readNumber(metrics?.distance_km);
      if (kilometers !== undefined && kilometers >= 0) return kilometers * 1000;
    }
    return 0;
  }

  private namedPlace(route: Prisma.JsonValue) {
    const object = jsonObject(route);
    if (!object) return null;
    const direct =
      readString(object.placeId) ?? readString(object.placeName) ?? readString(object.venueName);
    if (direct) return direct;
    const start = jsonObject(object.start);
    return readString(start?.placeId) ?? readString(start?.name) ?? readString(start?.address) ?? null;
  }

  private async accessiblePetIds(userId: string) {
    const links = await this.prisma.householdPet.findMany({
      where: {
        status: 'ACTIVE',
        household: { members: { some: { userId, status: 'ACTIVE' } } },
      },
      select: { petId: true },
    });
    return [...new Set(links.map((link) => link.petId))];
  }

  private activityWhere(userId: string, petId: string | undefined, before: Date | null) {
    return {
      AND: [
        this.households.householdActivityWhere(userId),
        ...(petId
          ? [
              {
                OR: [{ petId }, { petParticipants: { some: { petId } } }],
              } satisfies Prisma.ActivityWhereInput,
            ]
          : []),
        ...(before ? [{ startedAt: { lt: before } } satisfies Prisma.ActivityWhereInput] : []),
      ],
    } satisfies Prisma.ActivityWhereInput;
  }

  private careWhere(
    userId: string,
    accessiblePetIds: string[],
    petId: string | undefined,
    before: Date | null
  ) {
    return {
      AND: [
        {
          OR: [
            { userId },
            ...(accessiblePetIds.length
              ? [{ petId: { in: accessiblePetIds }, visibility: 'HOUSEHOLD' }]
              : []),
          ],
        },
        ...(petId ? [{ petId }] : []),
        ...(before ? [{ occurredAt: { lt: before } }] : []),
      ],
    } satisfies Prisma.CareEventWhereInput;
  }

  private mediaWhere(userId: string, petId: string | undefined, before: Date | null) {
    return {
      ownerId: userId,
      status: 'READY',
      ...(petId ? { petId } : {}),
      ...(before
        ? {
            OR: [
              { capturedAt: { lt: before } },
              { capturedAt: null, createdAt: { lt: before } },
            ],
          }
        : {}),
    } satisfies Prisma.MediaAssetWhereInput;
  }

  private async assertSourceAccessible(
    userId: string,
    sourceType: StorySourceType,
    sourceId: string
  ) {
    if (sourceType === 'ACTIVITY') {
      const activity = await this.prisma.activity.findFirst({
        where: { id: sourceId, AND: [this.households.householdActivityWhere(userId)] },
        select: { id: true },
      });
      if (!activity) throw new NotFoundException('Story source not found');
      return;
    }

    if (sourceType === 'MEDIA') {
      const media = await this.prisma.mediaAsset.findFirst({
        where: { id: sourceId, ownerId: userId, status: 'READY' },
        select: { id: true },
      });
      if (!media) throw new NotFoundException('Story source not found');
      return;
    }

    const accessiblePetIds = await this.accessiblePetIds(userId);
    const event = await this.prisma.careEvent.findFirst({
      where: {
        id: sourceId,
        OR: [
          { userId },
          ...(accessiblePetIds.length
            ? [{ petId: { in: accessiblePetIds }, visibility: 'HOUSEHOLD' }]
            : []),
        ],
      },
      select: { id: true },
    });
    if (!event) throw new NotFoundException('Story source not found');
  }

  private careTitle(eventType: string, pathway: string) {
    if (eventType === 'TRACKER_DAILY_ACTIVITY') return 'Daily movement summary';
    if (eventType === 'SAFE_OPT_OUT') return 'A good stop was the right choice';
    if (eventType.startsWith('QUEST_')) return `${humanize(pathway)} moment`;
    return humanize(eventType);
  }
}
