import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { TrackLocationDto } from './dto/track-location.dto';

@Injectable()
export class CoActivityService {
  constructor(private prisma: PrismaService) {}

  // Distance threshold for detecting co-activity (in meters)
  private readonly PROXIMITY_THRESHOLD = 50; // 50 meters
  private readonly TIME_WINDOW_MINUTES = 30; // 30 minutes

  /**
   * Track a user's location
   */
  async trackLocation(userId: string, dto: TrackLocationDto) {
    return this.prisma.locationPing.create({
      data: {
        userId,
        lat: dto.lat,
        lng: dto.lng,
        timestamp: new Date(dto.timestamp),
        activityType: dto.activityType,
      },
    });
  }

  /**
   * Get user's recent location history
   */
  async getUserLocations(userId: string, hours: number = 24) {
    const since = new Date();
    since.setHours(since.getHours() - hours);

    return this.prisma.locationPing.findMany({
      where: {
        userId,
        timestamp: {
          gte: since,
        },
      },
      orderBy: { timestamp: 'desc' },
    });
  }

  /**
   * Detect co-activity overlaps between two users
   * Returns instances where both users were in the same place at the same time
   */
  async detectOverlaps(userId1: string, userId2: string, hoursBack: number = 168) {
    const since = new Date();
    since.setHours(since.getHours() - hoursBack);

    // Get both users' location histories
    const [user1Locations, user2Locations] = await Promise.all([
      this.prisma.locationPing.findMany({
        where: { userId: userId1, timestamp: { gte: since } },
        orderBy: { timestamp: 'asc' },
      }),
      this.prisma.locationPing.findMany({
        where: { userId: userId2, timestamp: { gte: since } },
        orderBy: { timestamp: 'asc' },
      }),
    ]);

    const overlaps: any[] = [];

    // Check each location pair for proximity
    for (const loc1 of user1Locations) {
      for (const loc2 of user2Locations) {
        const distance = this.calculateDistance(
          loc1.lat,
          loc1.lng,
          loc2.lat,
          loc2.lng,
        );

        const timeDiff = Math.abs(
          new Date(loc1.timestamp).getTime() - new Date(loc2.timestamp).getTime(),
        ) / (1000 * 60); // minutes

        // If within proximity threshold and time window
        if (distance <= this.PROXIMITY_THRESHOLD && timeDiff <= this.TIME_WINDOW_MINUTES) {
          overlaps.push({
            location: { lat: loc1.lat, lng: loc1.lng },
            timestamp: loc1.timestamp,
            distance: Math.round(distance),
            timeDiff: Math.round(timeDiff),
            activityType: loc1.activityType || loc2.activityType,
          });
        }
      }
    }

    return {
      userId1,
      userId2,
      overlapCount: overlaps.length,
      overlaps,
    };
  }

  /**
   * Find potential co-activity matches for a user
   * Returns other users who were in same places at same times
   */
  async findPotentialMatches(userId: string, hoursBack: number = 168) {
    const since = new Date();
    since.setHours(since.getHours() - hoursBack);

    // Get user's location history
    const userLocations = await this.prisma.locationPing.findMany({
      where: { userId, timestamp: { gte: since } },
      orderBy: { timestamp: 'asc' },
    });

    if (userLocations.length === 0) {
      return [];
    }

    // Get all other users' locations in the same time window
    const otherLocations = await this.prisma.locationPing.findMany({
      where: {
        userId: { not: userId },
        timestamp: { gte: since },
      },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
      },
      orderBy: { timestamp: 'asc' },
    });

    // Group by user
    const locationsByUser = new Map<string, any[]>();
    for (const loc of otherLocations) {
      if (!locationsByUser.has(loc.userId)) {
        locationsByUser.set(loc.userId, []);
      }
      locationsByUser.get(loc.userId)!.push(loc);
    }

    // Calculate overlaps for each user
    const matches: any[] = [];

    for (const [otherUserId, otherLocs] of locationsByUser.entries()) {
      let overlapCount = 0;
      const overlaps: any[] = [];

      for (const userLoc of userLocations) {
        for (const otherLoc of otherLocs) {
          const distance = this.calculateDistance(
            userLoc.lat,
            userLoc.lng,
            otherLoc.lat,
            otherLoc.lng,
          );

          const timeDiff = Math.abs(
            new Date(userLoc.timestamp).getTime() - new Date(otherLoc.timestamp).getTime(),
          ) / (1000 * 60);

          if (distance <= this.PROXIMITY_THRESHOLD && timeDiff <= this.TIME_WINDOW_MINUTES) {
            overlapCount++;
            overlaps.push({
              location: { lat: userLoc.lat, lng: userLoc.lng },
              timestamp: userLoc.timestamp,
              distance: Math.round(distance),
            });
          }
        }
      }

      if (overlapCount > 0) {
        const otherUser = otherLocs[0].user;
        matches.push({
          user: otherUser,
          overlapCount,
          overlaps: overlaps.slice(0, 5), // Return first 5 overlaps
        });
      }
    }

    // Sort by overlap count descending
    matches.sort((a, b) => b.overlapCount - a.overlapCount);

    return matches;
  }

  /**
   * Calculate distance between two points using Haversine formula
   * Returns distance in meters
   */
  private calculateDistance(lat1: number, lng1: number, lat2: number, lng2: number): number {
    const R = 6371e3; // Earth's radius in meters
    const φ1 = (lat1 * Math.PI) / 180;
    const φ2 = (lat2 * Math.PI) / 180;
    const Δφ = ((lat2 - lat1) * Math.PI) / 180;
    const Δλ = ((lng2 - lng1) * Math.PI) / 180;

    const a =
      Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
      Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c; // Distance in meters
  }

  /**
   * Get co-activity statistics for a user
   */
  async getStats(userId: string) {
    const since = new Date();
    since.setDate(since.getDate() - 30); // Last 30 days

    const totalPings = await this.prisma.locationPing.count({
      where: { userId, timestamp: { gte: since } },
    });

    const matches = await this.findPotentialMatches(userId, 30 * 24);

    return {
      totalLocationPings: totalPings,
      potentialMatches: matches.length,
      topMatches: matches.slice(0, 5),
    };
  }
}
