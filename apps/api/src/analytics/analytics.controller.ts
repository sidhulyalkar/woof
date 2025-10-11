import { Controller, Get, Post, Body, Query, Param, UseGuards, Request } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiQuery, ApiBearerAuth } from '@nestjs/swagger';
import { AnalyticsService } from './analytics.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('analytics')
@Controller('analytics')
export class AnalyticsController {
  constructor(private readonly analyticsService: AnalyticsService) {}

  @Get('north-star')
  @ApiOperation({ summary: 'Get north star metrics for the specified timeframe' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'], description: 'Timeframe for metrics' })
  @ApiResponse({ status: 200, description: 'North star metrics retrieved successfully' })
  async getNorthStarMetrics(@Query('timeframe') timeframe?: string) {
    const minutes = this.getTimeframeMinutes(timeframe || '30d');
    return this.analyticsService.getNorthStarMetrics(minutes);
  }

  @Get('details')
  @ApiOperation({ summary: 'Get detailed analytics breakdown' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'], description: 'Timeframe for metrics' })
  @ApiResponse({ status: 200, description: 'Detailed metrics retrieved successfully' })
  async getDetailedMetrics(@Query('timeframe') timeframe?: string) {
    const minutes = this.getTimeframeMinutes(timeframe || '30d');
    return this.analyticsService.getDetailedMetrics(minutes);
  }

  @Post('telemetry')
  @ApiOperation({ summary: 'Record a telemetry event' })
  @ApiResponse({ status: 201, description: 'Telemetry event recorded' })
  async recordTelemetry(
    @Body() data: { userId?: string; source: string; event: string; metadata?: any },
  ) {
    return this.analyticsService.recordTelemetry(data);
  }

  @Get('events')
  @ApiOperation({ summary: 'Get event counts by type' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  @ApiResponse({ status: 200, description: 'Event counts retrieved' })
  async getEventCounts(@Query('timeframe') timeframe?: string) {
    const minutes = this.getTimeframeMinutes(timeframe || '30d');
    const since = new Date();
    since.setMinutes(since.getMinutes() - minutes);
    return this.analyticsService.getEventCounts(since);
  }

  @Get('users/active')
  @ApiOperation({ summary: 'Get active users count' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  @ApiResponse({ status: 200, description: 'Active users count' })
  async getActiveUsers(@Query('timeframe') timeframe?: string) {
    const minutes = this.getTimeframeMinutes(timeframe || '7d');
    const since = new Date();
    since.setMinutes(since.getMinutes() - minutes);
    return { activeUsers: await this.analyticsService.getActiveUsersCount(since) };
  }

  @Get('screens')
  @ApiOperation({ summary: 'Get screen view analytics' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  @ApiResponse({ status: 200, description: 'Screen view statistics' })
  async getScreenViews(@Query('timeframe') timeframe?: string) {
    const minutes = this.getTimeframeMinutes(timeframe || '7d');
    const since = new Date();
    since.setMinutes(since.getMinutes() - minutes);
    return this.analyticsService.getScreenViews(since);
  }

  @Get('users/:userId/activity')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get user activity timeline' })
  @ApiResponse({ status: 200, description: 'User activity timeline' })
  async getUserActivity(@Param('userId') userId: string, @Query('limit') limit?: number) {
    return this.analyticsService.getUserActivity(userId, limit ? parseInt(limit.toString()) : 50);
  }

  private getTimeframeMinutes(timeframe: string): number {
    switch (timeframe) {
      case '7d':
        return 7 * 24 * 60;
      case '30d':
        return 30 * 24 * 60;
      case '90d':
        return 90 * 24 * 60;
      default:
        return 30 * 24 * 60;
    }
  }
}
