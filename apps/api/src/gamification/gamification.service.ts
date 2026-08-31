import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class GamificationService {
  constructor(private prisma: PrismaService) {}

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

  async getUserBadges(userId: string) {
    return this.prisma.badgeAward.findMany({
      where: { userId },
      orderBy: { awardedAt: 'desc' },
    });
  }

  async getUserStreak(userId: string) {
    const streak = await this.prisma.weeklyStreak.findUnique({
      where: { userId },
    });

    if (!streak) {
      return { currentWeek: 0, lastActivityAt: null };
    }

    const lastActivity = new Date(streak.lastActivityAt);
    const today = new Date();
    const daysDiff = Math.floor((today.getTime() - lastActivity.getTime()) / (1000 * 60 * 60 * 24));

    if (daysDiff > 7) {
      return { ...streak, currentWeek: 0 };
    }

    return streak;
  }
}
