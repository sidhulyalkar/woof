import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { AwardPointsDto } from './dto/award-points.dto';
import { AwardBadgeDto, BadgeType } from './dto/award-badge.dto';
import { UpdateStreakDto } from './dto/update-streak.dto';

@Injectable()
export class GamificationService {
  constructor(private prisma: PrismaService) {}

  async awardPoints(dto: AwardPointsDto) {
    const user = await this.prisma.user.findUnique({
      where: { id: dto.userId },
    });

    if (!user) {
      throw new NotFoundException(`User ${dto.userId} not found`);
    }

    const transaction = await this.prisma.pointTransaction.create({
      data: {
        userId: dto.userId,
        points: dto.points,
        reason: dto.reason,
        relatedEntityId: dto.relatedEntityId,
      },
    });

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

  async getPointTransactions(userId: string) {
    return this.prisma.pointTransaction.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
    });
  }

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

  async awardBadge(dto: AwardBadgeDto) {
    const existingBadge = await this.prisma.badgeAward.findUnique({
      where: {
        userId_badgeType: {
          userId: dto.userId,
          badgeType: dto.badgeType,
        },
      },
    });

    if (existingBadge) {
      return existingBadge;
    }

    return this.prisma.badgeAward.create({
      data: {
        userId: dto.userId,
        badgeType: dto.badgeType,
      },
    });
  }

  async getUserBadges(userId: string) {
    return this.prisma.badgeAward.findMany({
      where: { userId },
      orderBy: { awardedAt: 'desc' },
    });
  }

  async updateStreak(dto: UpdateStreakDto) {
    const activityDate = new Date(dto.activityDate);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    let streak = await this.prisma.weeklyStreak.findUnique({
      where: { userId: dto.userId },
    });

    if (!streak) {
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

    if (daysDiff === 0) {
      return streak;
    }

    if (daysDiff <= 7) {
      const lastWeekStart = this.getWeekStart(lastActivity);
      const currentWeekStart = this.getWeekStart(activityDate);

      if (currentWeekStart > lastWeekStart) {
        streak = await this.prisma.weeklyStreak.update({
          where: { userId: dto.userId },
          data: {
            currentWeek: { increment: 1 },
            lastActivityAt: activityDate,
          },
        });

        if (streak.currentWeek >= 4) {
          await this.awardBadge({
            userId: dto.userId,
            badgeType: BadgeType.STREAK_MASTER,
          });
        }
      } else {
        streak = await this.prisma.weeklyStreak.update({
          where: { userId: dto.userId },
          data: { lastActivityAt: activityDate },
        });
      }
    } else {
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

  async getUserStreak(userId: string) {
    let streak = await this.prisma.weeklyStreak.findUnique({
      where: { userId },
    });

    if (!streak) {
      return { currentWeek: 0, lastActivityAt: null };
    }

    const lastActivity = new Date(streak.lastActivityAt);
    const today = new Date();
    const daysDiff = Math.floor((today.getTime() - lastActivity.getTime()) / (1000 * 60 * 60 * 24));

    if (daysDiff > 7) {
      streak = await this.prisma.weeklyStreak.update({
        where: { userId },
        data: { currentWeek: 0 },
      });
    }

    return streak;
  }

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

  private getWeekStart(date: Date): Date {
    const d = new Date(date);
    const day = d.getDay();
    const diff = d.getDate() - day + (day === 0 ? -6 : 1);
    return new Date(d.setDate(diff));
  }
}
