import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { AwardPointsDto } from './dto/award-points.dto';
import { AwardBadgeDto, BadgeType } from './dto/award-badge.dto';
import { UpdateStreakDto } from './dto/update-streak.dto';

@Injectable()
export class GamificationService {
  constructor(private prisma: PrismaService) {}

  /**
   * Award points to a user and create a transaction record
   */
  async awardPoints(dto: AwardPointsDto) {
    // Verify user exists
    const user = await this.prisma.user.findUnique({
      where: { id: dto.userId },
    });

    if (!user) {
      throw new NotFoundException(`User ${dto.userId} not found`);
    }

    // Create point transaction
    const transaction = await this.prisma.pointTransaction.create({
      data: {
        userId: dto.userId,
        points: dto.points,
        reason: dto.reason,
        relatedEntityId: dto.relatedEntityId,
      },
    });

    // Update user's total points
    await this.prisma.user.update({
      where: { id: dto.userId },
      data: {
        totalPoints: {
          increment: dto.points,
        },
      },
    });

    return transaction;
  }

  /**
   * Get all point transactions for a user
   */
  async getPointTransactions(userId: string) {
    return this.prisma.pointTransaction.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
    });
  }

  /**
   * Get user's total points
   */
  async getUserPoints(userId: string) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { totalPoints: true },
    });

    if (!user) {
      throw new NotFoundException(`User ${userId} not found`);
    }

    return { totalPoints: user.totalPoints || 0 };
  }

  /**
   * Award a badge to a user (idempotent - won't duplicate)
   */
  async awardBadge(dto: AwardBadgeDto) {
    // Check if user already has this badge
    const existingBadge = await this.prisma.badgeAward.findUnique({
      where: {
        userId_badgeType: {
          userId: dto.userId,
          badgeType: dto.badgeType,
        },
      },
    });

    if (existingBadge) {
      return existingBadge; // Idempotent - don't award duplicate badges
    }

    // Award the badge
    return this.prisma.badgeAward.create({
      data: {
        userId: dto.userId,
        badgeType: dto.badgeType,
      },
    });
  }

  /**
   * Get all badges for a user
   */
  async getUserBadges(userId: string) {
    return this.prisma.badgeAward.findMany({
      where: { userId },
      orderBy: { awardedAt: 'desc' },
    });
  }

  /**
   * Update user's activity streak
   */
  async updateStreak(dto: UpdateStreakDto) {
    const activityDate = new Date(dto.activityDate);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Get or create streak record
    let streak = await this.prisma.weeklyStreak.findUnique({
      where: { userId: dto.userId },
    });

    if (!streak) {
      // First activity - create new streak
      streak = await this.prisma.weeklyStreak.create({
        data: {
          userId: dto.userId,
          currentWeek: 1,
          lastActivityAt: activityDate,
        },
      });
      return streak;
    }

    const lastActivity = new Date(streak.lastActivityAt);
    lastActivity.setHours(0, 0, 0, 0);

    const daysDiff = Math.floor((today.getTime() - lastActivity.getTime()) / (1000 * 60 * 60 * 24));

    // Same day - no update needed
    if (daysDiff === 0) {
      return streak;
    }

    // Within 7 days - increment week if crossed into new week
    if (daysDiff <= 7) {
      const lastWeekStart = this.getWeekStart(lastActivity);
      const currentWeekStart = this.getWeekStart(activityDate);

      if (currentWeekStart > lastWeekStart) {
        // Moved to next week
        streak = await this.prisma.weeklyStreak.update({
          where: { userId: dto.userId },
          data: {
            currentWeek: { increment: 1 },
            lastActivityAt: activityDate,
          },
        });

        // Award streak master badge if 4+ weeks
        if (streak.currentWeek >= 4) {
          await this.awardBadge({
            userId: dto.userId,
            badgeType: BadgeType.STREAK_MASTER,
          });
        }
      } else {
        // Same week - just update last activity
        streak = await this.prisma.weeklyStreak.update({
          where: { userId: dto.userId },
          data: { lastActivityAt: activityDate },
        });
      }
    } else {
      // More than 7 days - streak broken, reset to week 1
      streak = await this.prisma.weeklyStreak.update({
        where: { userId: dto.userId },
        data: {
          currentWeek: 1,
          lastActivityAt: activityDate,
        },
      });
    }

    return streak;
  }

  /**
   * Get user's current streak
   */
  async getUserStreak(userId: string) {
    let streak = await this.prisma.weeklyStreak.findUnique({
      where: { userId },
    });

    if (!streak) {
      return { currentWeek: 0, lastActivityAt: null };
    }

    // Check if streak is still valid (last activity within 7 days)
    const lastActivity = new Date(streak.lastActivityAt);
    const today = new Date();
    const daysDiff = Math.floor((today.getTime() - lastActivity.getTime()) / (1000 * 60 * 60 * 24));

    if (daysDiff > 7) {
      // Streak expired - reset it
      streak = await this.prisma.weeklyStreak.update({
        where: { userId },
        data: { currentWeek: 0 },
      });
    }

    return streak;
  }

  /**
   * Get leaderboard (top users by points)
   */
  async getLeaderboard(limit: number = 20) {
    const users = await this.prisma.user.findMany({
      orderBy: { totalPoints: 'desc' },
      take: limit,
      select: {
        id: true,
        handle: true,
        avatarUrl: true,
        totalPoints: true,
      },
    });

    return users.map((user, index) => ({
      rank: index + 1,
      ...user,
    }));
  }

  /**
   * Helper: Get the start of the week (Monday) for a given date
   */
  private getWeekStart(date: Date): Date {
    const d = new Date(date);
    const day = d.getDay();
    const diff = d.getDate() - day + (day === 0 ? -6 : 1); // Adjust when Sunday
    return new Date(d.setDate(diff));
  }
}
