import { Controller, Get, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { GamificationService } from './gamification.service';

@ApiTags('gamification')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('gamification')
export class GamificationController {
  constructor(private readonly gamificationService: GamificationService) {}

  @Get('me/summary')
  @ApiOperation({
    summary: 'Get the legacy reward summary for the current user',
    description:
      'Read-only compatibility endpoint. All new rewards are issued by trusted domain events through the Adventure System.',
  })
  async getMySummary(@Request() req: any) {
    const userId = req.user.sub;
    const [points, badges, streak] = await Promise.all([
      this.gamificationService.getUserPoints(userId),
      this.gamificationService.getUserBadges(userId),
      this.gamificationService.getUserStreak(userId),
    ]);

    return {
      points: points.totalPoints,
      badges: badges.map((badge: any) => badge.badgeType),
      badgeCount: badges.length,
      streak: streak.currentWeek,
      lastActivity: streak.lastActivityAt,
      deprecated: true,
      replacement: '/adventure/me',
    };
  }
}
