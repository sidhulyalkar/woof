import { Controller, Get, Post, Body, Param, Query, UseGuards, Request } from '@nestjs/common';
import { ApiBearerAuth, ApiTags, ApiOperation, ApiResponse, ApiQuery } from '@nestjs/swagger';
import { GamificationService } from './gamification.service';
import { AwardPointsDto } from './dto/award-points.dto';
import { AwardBadgeDto } from './dto/award-badge.dto';
import { UpdateStreakDto } from './dto/update-streak.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

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
  @ApiResponse({ status: 200, description: 'Point transactions retrieved' })
  async getPointTransactions(@Param('userId') userId: string) {
    return this.gamificationService.getPointTransactions(userId);
  }

  @Post('badges')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Award a badge to a user' })
  @ApiResponse({ status: 201, description: 'Badge awarded successfully' })
  async awardBadge(@Body() awardBadgeDto: AwardBadgeDto) {
    return this.gamificationService.awardBadge(awardBadgeDto);
  }

  @Get('badges/:userId')
  @ApiOperation({ summary: 'Get all badges for a user' })
  @ApiResponse({ status: 200, description: 'User badges retrieved' })
  async getUserBadges(@Param('userId') userId: string) {
    return this.gamificationService.getUserBadges(userId);
  }

  @Post('streaks')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Update user activity streak' })
  @ApiResponse({ status: 201, description: 'Streak updated successfully' })
  async updateStreak(@Body() updateStreakDto: UpdateStreakDto) {
    return this.gamificationService.updateStreak(updateStreakDto);
  }

  @Get('streaks/:userId')
  @ApiOperation({ summary: 'Get current streak for a user' })
  @ApiResponse({ status: 200, description: 'User streak retrieved' })
  async getUserStreak(@Param('userId') userId: string) {
    return this.gamificationService.getUserStreak(userId);
  }

  @Get('leaderboard')
  @ApiOperation({ summary: 'Get points leaderboard' })
  @ApiQuery({ name: 'limit', required: false, description: 'Number of top users to return', example: 20 })
  @ApiResponse({ status: 200, description: 'Leaderboard retrieved' })
  async getLeaderboard(@Query('limit') limit?: string) {
    const limitNum = limit ? parseInt(limit, 10) : 20;
    return this.gamificationService.getLeaderboard(limitNum);
  }

  @Get('me/summary')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get gamification summary for current user' })
  @ApiResponse({ status: 200, description: 'User gamification summary' })
  async getMySummary(@Request() req: any) {
    const userId = req.user.id;

    const [points, badges, streak] = await Promise.all([
      this.gamificationService.getUserPoints(userId),
      this.gamificationService.getUserBadges(userId),
      this.gamificationService.getUserStreak(userId),
    ]);

    return {
      points: points.totalPoints,
      badges: badges.map((b: any) => b.badgeType),
      badgeCount: badges.length,
      streak: streak.currentWeek,
      lastActivity: streak.lastActivityAt,
    };
  }
}
