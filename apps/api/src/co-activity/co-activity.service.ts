import { BadRequestException, ForbiddenException, Injectable } from '@nestjs/common';
import { PrivacyService } from '../privacy/privacy.service';
import { PrismaService } from '../prisma/prisma.service';
import { TrackLocationDto } from './dto/track-location.dto';

@Injectable()
export class CoActivityService {
  private readonly PROXIMITY_THRESHOLD_M = 50;
  private readonly TIME_WINDOW_MINUTES = 30;
  private readonly MAX_QUERY_HOURS = 24;
  private readonly MAX_CLOCK_SKEW_MINUTES = 5;

  constructor(
    private readonly prisma: PrismaService,
    private readonly privacyService: PrivacyService,
  ) {}

  async trackLocation(userId: string, dto: TrackLocationDto) {
    const preferences = await this.privacyService.requirePreciseLocation(userId);
    const observedAt = dto.timestamp ? new Date(dto.timestamp) : new Date();
    const skewMinutes = Math.abs(Date.now() - observedAt.getTime()) / (60 * 1000);
    if (!Number.isFinite(observedAt.getTime()) || skewMinutes > this.MAX_CLOCK_SKEW_MINUTES) {
      throw new BadRequestException('Location timestamp must be within five minutes of server time');
    }

    await this.privacyService.pruneLocationHistory(userId, preferences.locationRetentionHours);
    const ping = await this.prisma.locationPing.create({
      data: {
        userId,
        lat: dto.lat,
        lng: dto.lng,
        timestamp: observedAt,
        activityType: dto.activityType,
      },
      select: { id: true, timestamp: true },
    });

    return {
      accepted: true,
      id: ping.id,
      observedAt: ping.timestamp,
      retainedUntil: new Date(
        ping.timestamp.getTime() + preferences.locationRetentionHours * 60 * 60 * 1000,
      ),
    };
  }

  async getLocationSummary(userId: string) {
    return this.privacyService.getLocationSummary(userId);
  }

  async detectOverlaps(userId1: string, userId2: string, hoursBack = 12) {
    if (userId1 === userId2) throw new BadRequestException('Choose another member');
    if (!(await this.privacyService.bothAllowProximity(userId1, userId2))) {
      throw new ForbiddenException(
        'Co-activity requires both members to opt into proximity suggestions and neither member to be blocked.',
      );
    }

    const hours = this.safeHours(hoursBack);
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);
    const [user1Locations, user2Locations] = await Promise.all([
      this.prisma.locationPing.findMany({
        where: { userId: userId1, timestamp: { gte: since } },
        orderBy: { timestamp: 'asc' },
        take: 500,
      }),
      this.prisma.locationPing.findMany({
        where: { userId: userId2, timestamp: { gte: since } },
        orderBy: { timestamp: 'asc' },
        take: 500,
      }),
    ]);

    const overlaps = this.findOverlapWindows(user1Locations, user2Locations);
    const latest = overlaps.at(-1);
    return {
      userId: userId2,
      overlapCount: overlaps.length,
      proximity: overlaps.length > 0 ? 'WITHIN_50M' : 'NONE_DETECTED',
      latestOverlapAt: latest?.timestamp ?? null,
      queryHours: hours,
      coordinatesDisclosed: false,
    };
  }

  async findPotentialMatches(userId: string, hoursBack = 12) {
    if (!(await this.privacyService.canUseProximity(userId))) {
      throw new ForbiddenException('Enable precise location and proximity suggestions first');
    }

    const preferences = await this.privacyService.getPreferences(userId);
    await this.privacyService.pruneLocationHistory(userId, preferences.locationRetentionHours);
    const hours = Math.min(this.safeHours(hoursBack), preferences.locationRetentionHours);
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);

    const [userLocations, otherLocations, blocks] = await Promise.all([
      this.prisma.locationPing.findMany({
        where: { userId, timestamp: { gte: since } },
        orderBy: { timestamp: 'asc' },
        take: 500,
      }),
      this.prisma.locationPing.findMany({
        where: { userId: { not: userId }, timestamp: { gte: since } },
        include: {
          user: { select: { id: true, handle: true, avatarUrl: true, visibility: true } },
        },
        orderBy: { timestamp: 'asc' },
        take: 5000,
      }),
      this.prisma.blockedUser.findMany({
        where: { OR: [{ userId }, { blockedId: userId }] },
        select: { userId: true, blockedId: true },
      }),
    ]);
    if (userLocations.length === 0) return [];

    const blockedIds = new Set(
      blocks.map((block) => (block.userId === userId ? block.blockedId : block.userId)),
    );
    const otherUserIds = [...new Set(otherLocations.map((location) => location.userId))];
    const privacy = await this.privacyService.getPreferencesForUsers(otherUserIds);
    const locationsByUser = new Map<string, typeof otherLocations>();

    for (const location of otherLocations) {
      const prefs = privacy.get(location.userId);
      if (
        blockedIds.has(location.userId) ||
        location.user.visibility === 'PRIVATE' ||
        !prefs?.preciseLocation ||
        !prefs.proximitySuggestions
      ) {
        continue;
      }
      const existing = locationsByUser.get(location.userId) ?? [];
      existing.push(location);
      locationsByUser.set(location.userId, existing);
    }

    const matches = [];
    for (const [otherUserId, otherUserLocations] of locationsByUser.entries()) {
      const overlaps = this.findOverlapWindows(userLocations, otherUserLocations);
      if (overlaps.length === 0) continue;
      matches.push({
        user: otherUserLocations[0].user,
        overlapCount: overlaps.length,
        proximity: 'WITHIN_50M',
        latestOverlapAt: overlaps.at(-1)?.timestamp ?? null,
        coordinatesDisclosed: false,
      });
    }

    return matches.sort((a, b) => b.overlapCount - a.overlapCount).slice(0, 20);
  }

  async getStats(userId: string) {
    const summary = await this.privacyService.getLocationSummary(userId);
    if (!summary.preferences.proximitySuggestions) {
      return {
        ...summary,
        potentialMatches: 0,
        proximitySuggestionsEnabled: false,
      };
    }
    const matches = await this.findPotentialMatches(
      userId,
      summary.preferences.locationRetentionHours,
    );
    return {
      ...summary,
      potentialMatches: matches.length,
      proximitySuggestionsEnabled: true,
    };
  }

  private findOverlapWindows(
    user1Locations: Array<{ lat: number; lng: number; timestamp: Date }>,
    user2Locations: Array<{ lat: number; lng: number; timestamp: Date }>,
  ) {
    const overlaps: Array<{ timestamp: Date }> = [];
    let lastRecordedAt = 0;

    for (const loc1 of user1Locations) {
      for (const loc2 of user2Locations) {
        const timeDiffMinutes =
          Math.abs(loc1.timestamp.getTime() - loc2.timestamp.getTime()) / (60 * 1000);
        if (timeDiffMinutes > this.TIME_WINDOW_MINUTES) continue;
        const distance = this.calculateDistance(loc1.lat, loc1.lng, loc2.lat, loc2.lng);
        if (distance > this.PROXIMITY_THRESHOLD_M) continue;

        const midpointTime = Math.round((loc1.timestamp.getTime() + loc2.timestamp.getTime()) / 2);
        // Collapse dense GPS samples into one privacy-preserving encounter window.
        if (midpointTime - lastRecordedAt < this.TIME_WINDOW_MINUTES * 60 * 1000) continue;
        overlaps.push({ timestamp: new Date(midpointTime) });
        lastRecordedAt = midpointTime;
      }
    }
    return overlaps;
  }

  private safeHours(value: number) {
    return Math.max(1, Math.min(Number(value) || 12, this.MAX_QUERY_HOURS));
  }

  private calculateDistance(lat1: number, lng1: number, lat2: number, lng2: number): number {
    const earthRadiusM = 6371e3;
    const phi1 = (lat1 * Math.PI) / 180;
    const phi2 = (lat2 * Math.PI) / 180;
    const deltaPhi = ((lat2 - lat1) * Math.PI) / 180;
    const deltaLambda = ((lng2 - lng1) * Math.PI) / 180;
    const a =
      Math.sin(deltaPhi / 2) ** 2 +
      Math.cos(phi1) * Math.cos(phi2) * Math.sin(deltaLambda / 2) ** 2;
    return earthRadiusM * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }
}
