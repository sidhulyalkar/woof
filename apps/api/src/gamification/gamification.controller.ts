import { Body, Controller, Get, Param, Post, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiQuery, ApiResponse, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AwardBadgeDto } from './dto/award-badge.dto';
import { AwardPointsDto } from './dto/award-points.dto';
import { UpdateStreakDto } from './dto/update-streak.dto';
import { GamificationService } from './gamification.service';

@ApiTags('gamification')
@Controller('gamification')
export class GamificationController {
  constructor(private readonly gamificationService: GamificationService) {}

  @Post('points')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Award points to a user' })
  @ApiResponse({ status: 201, description: 'Points awarded successfully' })
  async awardPoints(@Body() awardPointsDto: AwardPointsDto) {
    return this.gamificationService.awardPoints(awardPointsDto);
  }

  @Get('points/:userId')
  @ApiOperation({ summary: 'Get total points for a user' })
  @ApiResponse({ status: 200, description: 'User points retrieved' })
  async getUserPoints(@Param('userId') userId: string) {
    return this.gamificationService.getUserPoints(userId);
  }

  @Get('points/:userId/transactions')
  @ApiOperation({ summary: 'Get point transaction history for a user' })
  async getPointTransactions(@Param('userId') userId: string) {
    return this.gamificationService.getPointTransactions(userId);
  }

  @Post('badges')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Award a badge to a user' })
  async awardBadge(@Body() awardBadgeDto: AwardBadgeDto) {
    return this.gamificationService.awardBadge(awardBadgeDto);
  }

  @Get('badges/:userId')
  @ApiOperation({ summary: 'Get all badges for a user' })
  async getUserBadges(@Param('userId') userId: string) {
    return this.gamificationService.getUserBadges(userId);
  }

  @Post('streaks')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Update user activity streak' })
  async updateStreak(@Body() updateStreakDto: UpdateStreakDto) {
    return this.gamificationService.updateStreak(updateStreakDto);
  }

  @Get('streaks/:userId')
  @ApiOperation({ summary: 'Get current streak for a user' })
  async getUserStreak(@Param('userId') userId: string) {
    return this.gamificationService.getUserStreak(userId);
  }

  @Get('leaderboard')
  @ApiOperation({ summary: 'Get points leaderboard' })
  @ApiQuery({ name: 'limit', required: false, description: 'Number of top users to return', example: 20 })
  async getLeaderboard(@Query('limit') limit?: string) {
    const parsed = limit ? Number.parseInt(limit, 10) : 20;
    const safeLimit = Number.isFinite(parsed) ? Math.max(1, Math.min(parsed, 100)) : 20;
    return this.gamificationService.getLeaderboard(safeLimit);
  }

  @Get('me/summary')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get gamification summary for current user' })
  async getMySummary(@Request() req: AuthenticatedRequest) {
    const userId = req.user.sub;

    const [points, badges, streak] = await Promise.all([
      this.gamificationService.getUserPoints(userId),
      this.gamificationService.getUserBadges(userId),
      this.gamificationService.getUserStreak(userId),
    ]);

    return {
      points: points.totalPoints,
      badges: badges.map((badge) => badge.badgeType),
      badgeCount: badges.length,
      streak: streak.currentWeek,
      lastActivity: streak.lastActivityAt,
    };
  }
}
