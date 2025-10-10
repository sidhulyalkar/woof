import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class AnalyticsService {
  constructor(private prisma: PrismaService) {}

  /**
   * Calculate North Star Metrics for the specified timeframe
   * 1. Meetup Conversion Rate = confirmed meetups / unique match chats
   * 2. 7D Retention = % users returning within 7 days
   * 3. Data Yield per User = labeled interactions + verified outcomes
   * 4. Service Intent Rate = service taps to book / MAU
   */
  async getNorthStarMetrics(timeframeMinutes: number) {
    const since = new Date();
    since.setMinutes(since.getMinutes() - timeframeMinutes);

    const [
      meetupConversion,
      retention7D,
      dataYield,
      serviceIntent,
      userStats,
      meetupStats,
    ] = await Promise.all([
      this.calculateMeetupConversion(since),
      this.calculate7DayRetention(since),
      this.calculateDataYieldPerUser(since),
      this.calculateServiceIntentRate(since),
      this.getUserStats(since),
      this.getMeetupStats(since),
    ]);

    return {
      meetupConversionRate: meetupConversion.rate,
      meetupConversionRateTrend: meetupConversion.trend,
      retention7Day: retention7D.rate,
      retention7DayTrend: retention7D.trend,
      dataYieldPerUser: dataYield.yieldPerUser,
      dataYieldPerUserTrend: dataYield.trend,
      serviceIntentRate: serviceIntent.rate,
      serviceIntentRateTrend: serviceIntent.trend,
      totalUsers: userStats.total,
      activeUsers: userStats.active,
      totalMeetups: meetupStats.total,
      completedMeetups: meetupStats.completed,
    };
  }

  /**
   * Get detailed metrics for analytics breakdown
   */
  async getDetailedMetrics(timeframeMinutes: number) {
    const since = new Date();
    since.setMinutes(since.getMinutes() - timeframeMinutes);

    // Meetup funnel
    const totalMatches = await this.prisma.compatibilityEdge.count({
      where: {
        status: 'matched',
        createdAt: { gte: since },
      },
    });

    const uniqueMatchChats = await this.prisma.compatibilityEdge.count({
      where: {
        status: 'matched',
        createdAt: { gte: since },
        // Assume if they have any activity, they chatted
      },
    });

    // Note: Using user table as proxy for proposals since MeetupProposal table may not exist yet
    const meetupsProposed = 0; // Will be populated when meetupProposal table is available
    const meetupsConfirmed = 0;
    const meetupsCompleted = 0;

    // Service intents
    const serviceIntents = await this.prisma.serviceIntent.groupBy({
      by: ['action'],
      _count: true,
      where: { createdAt: { gte: since } },
    });

    const totalIntents = serviceIntents.reduce((sum, group) => sum + group._count, 0);
    const tapBook = serviceIntents.find(g => g.action === 'tap_book')?._count || 0;
    const conversions = await this.prisma.serviceIntent.count({
      where: {
        action: 'tap_book',
        conversionFollowup: true,
        createdAt: { gte: since },
      },
    });

    // Event feedback
    const eventFeedback = await this.prisma.eventFeedback.aggregate({
      where: { createdAt: { gte: since } },
      _count: true,
      _avg: {
        vibeScore: true,
        petDensity: true,
        venueQuality: true,
      },
    });

    // Avg feedback per user
    const totalUsers = await this.prisma.user.count({
      where: { createdAt: { lte: since } },
    });

    const totalFeedbackItems = meetupsCompleted + (eventFeedback._count || 0);
    const avgFeedbackPerUser = totalUsers > 0 ? totalFeedbackItems / totalUsers : 0;

    return {
      totalMatches,
      uniqueMatchChats,
      meetupsProposed,
      meetupsConfirmed,
      meetupsCompleted,
      avgFeedbackPerUser,
      serviceIntents: {
        total: totalIntents,
        tapBook,
        conversions,
      },
      eventsFeedback: {
        total: eventFeedback._count || 0,
        avgVibeScore: eventFeedback._avg.vibeScore || 0,
        avgPetDensity: eventFeedback._avg.petDensity || 0,
        avgVenueQuality: eventFeedback._avg.venueQuality || 0,
      },
    };
  }

  /**
   * Private helper methods for metric calculations
   */

  private async calculateMeetupConversion(since: Date) {
    // Unique match chats (approximated by matched edges)
    const uniqueMatchChats = await this.prisma.compatibilityEdge.count({
      where: {
        status: 'matched',
        createdAt: { gte: since },
      },
    });

    // Confirmed meetups (will need meetupProposal table when available)
    const confirmedMeetups = 0; // Placeholder

    const rate = uniqueMatchChats > 0 ? (confirmedMeetups / uniqueMatchChats) * 100 : 0;

    // Calculate trend (compare to previous period)
    const previousPeriod = new Date(since);
    const periodLength = Date.now() - since.getTime();
    previousPeriod.setTime(previousPeriod.getTime() - periodLength);

    const previousChats = await this.prisma.compatibilityEdge.count({
      where: {
        status: 'matched',
        createdAt: { gte: previousPeriod, lt: since },
      },
    });

    const previousRate = previousChats > 0 ? (0 / previousChats) * 100 : 0;
    const trend = previousRate > 0 ? ((rate - previousRate) / previousRate) * 100 : 0;

    return { rate, trend };
  }

  private async calculate7DayRetention(since: Date) {
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

    // Users who joined before 7 days ago
    const cohortUsers = await this.prisma.user.findMany({
      where: {
        createdAt: { gte: sevenDaysAgo, lt: since },
      },
      select: { id: true },
    });

    if (cohortUsers.length === 0) {
      return { rate: 0, trend: 0 };
    }

    // Count how many returned (had any activity in last 7 days)
    const activeUsers = await this.prisma.user.count({
      where: {
        id: { in: cohortUsers.map(u => u.id) },
        updatedAt: { gte: sevenDaysAgo },
      },
    });

    const rate = (activeUsers / cohortUsers.length) * 100;

    return { rate, trend: 0 }; // Simplified - not calculating trend for now
  }

  private async calculateDataYieldPerUser(since: Date) {
    const totalUsers = await this.prisma.user.count({
      where: { createdAt: { lte: since } },
    });

    if (totalUsers === 0) {
      return { yieldPerUser: 0, trend: 0 };
    }

    // Count labeled interactions
    const [serviceIntents, eventFeedback] = await Promise.all([
      this.prisma.serviceIntent.count({ where: { createdAt: { gte: since } } }),
      this.prisma.eventFeedback.count({ where: { createdAt: { gte: since } } }),
    ]);

    const totalInteractions = serviceIntents + eventFeedback;
    const yieldPerUser = totalInteractions / totalUsers;

    return { yieldPerUser, trend: 0 };
  }

  private async calculateServiceIntentRate(since: Date) {
    const activeUsers = await this.prisma.user.count({
      where: { updatedAt: { gte: since } },
    });

    if (activeUsers === 0) {
      return { rate: 0, trend: 0 };
    }

    const serviceTaps = await this.prisma.serviceIntent.count({
      where: {
        action: 'tap_book',
        createdAt: { gte: since },
      },
    });

    const rate = (serviceTaps / activeUsers) * 100;

    return { rate, trend: 0 };
  }

  private async getUserStats(since: Date) {
    const [total, active] = await Promise.all([
      this.prisma.user.count(),
      this.prisma.user.count({
        where: { updatedAt: { gte: since } },
      }),
    ]);

    return { total, active };
  }

  private async getMeetupStats(since: Date) {
    // Placeholder - will use meetupProposal table when available
    return {
      total: 0,
      completed: 0,
    };
  }

  /**
   * Record telemetry event
   */
  async recordTelemetry(data: {
    userId?: string;
    source: string;
    event: string;
    metadata?: any;
  }) {
    return this.prisma.telemetry.create({
      data: {
        userId: data.userId,
        source: data.source,
        event: data.event,
        metadata: data.metadata || {},
      },
    });
  }

  /**
   * Get event counts by type
   */
  async getEventCounts(since: Date) {
    const events = await this.prisma.telemetry.groupBy({
      by: ['event'],
      _count: true,
      where: { createdAt: { gte: since } },
      orderBy: { _count: { event: 'desc' } },
    });

    return events.map(e => ({
      event: e.event,
      count: e._count,
    }));
  }

  /**
   * Get user activity timeline
   */
  async getUserActivity(userId: string, limit = 50) {
    return this.prisma.telemetry.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: limit,
    });
  }

  /**
   * Get active users count (opened app in timeframe)
   */
  async getActiveUsersCount(since: Date) {
    const result = await this.prisma.telemetry.groupBy({
      by: ['userId'],
      where: {
        event: 'APP_OPEN',
        createdAt: { gte: since },
      },
    });

    return result.length;
  }

  /**
   * Get screen view analytics
   */
  async getScreenViews(since: Date) {
    const screens = await this.prisma.telemetry.groupBy({
      by: ['metadata'],
      _count: true,
      where: {
        event: 'SCREEN_VIEW',
        createdAt: { gte: since },
      },
    });

    return screens.map(s => ({
      screen: (s.metadata as any)?.screen || 'unknown',
      views: s._count,
    }));
  }
}
