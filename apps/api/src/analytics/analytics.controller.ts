import {
  Body,
  Controller,
  ForbiddenException,
  Get,
  Param,
  Post,
  Query,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiQuery, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AnalyticsService } from './analytics.service';

@ApiTags('analytics')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('analytics')
export class AnalyticsController {
  constructor(private readonly analyticsService: AnalyticsService) {}

  @Get('north-star')
  @ApiOperation({ summary: 'Get outcome-focused beta metrics' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  getNorthStarMetrics(@Query('timeframe') timeframe?: string) {
    return this.analyticsService.getNorthStarMetrics(this.getTimeframeMinutes(timeframe || '30d'));
  }

  @Get('details')
  @ApiOperation({ summary: 'Get relationship funnel and compatibility calibration details' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  getDetailedMetrics(@Query('timeframe') timeframe?: string) {
    return this.analyticsService.getDetailedMetrics(this.getTimeframeMinutes(timeframe || '30d'));
  }

  @Get('compatibility-calibration')
  @ApiOperation({ summary: 'Get compatibility fallback and future-outcome calibration dashboard data' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  getCompatibilityCalibration(@Query('timeframe') timeframe?: string) {
    const since = new Date(Date.now() - this.getTimeframeMinutes(timeframe || '30d') * 60 * 1000);
    return this.analyticsService.getCompatibilityCalibration(since);
  }

  @Post('telemetry')
  @ApiOperation({ summary: 'Record actor-bound product telemetry' })
  recordTelemetry(
    @Request() req: AuthenticatedRequest,
    @Body() data: { source: string; event: string; metadata?: unknown },
  ) {
    return this.analyticsService.recordTelemetry({
      userId: req.user.sub,
      source: data.source,
      event: data.event,
      metadata: data.metadata,
    });
  }

  @Get('events')
  @ApiOperation({ summary: 'Get product event counts' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  getEventCounts(@Query('timeframe') timeframe?: string) {
    const since = new Date(Date.now() - this.getTimeframeMinutes(timeframe || '30d') * 60 * 1000);
    return this.analyticsService.getEventCounts(since);
  }

  @Get('users/active')
  @ApiOperation({ summary: 'Get active user count' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  async getActiveUsers(@Query('timeframe') timeframe?: string) {
    const since = new Date(Date.now() - this.getTimeframeMinutes(timeframe || '7d') * 60 * 1000);
    return { activeUsers: await this.analyticsService.getActiveUsersCount(since) };
  }

  @Get('screens')
  @ApiOperation({ summary: 'Get screen view analytics' })
  @ApiQuery({ name: 'timeframe', required: false, enum: ['7d', '30d', '90d'] })
  getScreenViews(@Query('timeframe') timeframe?: string) {
    const since = new Date(Date.now() - this.getTimeframeMinutes(timeframe || '7d') * 60 * 1000);
    return this.analyticsService.getScreenViews(since);
  }

  @Get('users/:userId/activity')
  @ApiOperation({ summary: 'Get the signed-in user telemetry timeline' })
  getUserActivity(
    @Request() req: AuthenticatedRequest,
    @Param('userId') userId: string,
    @Query('limit') limit?: number,
  ) {
    if (userId !== req.user.sub) {
      throw new ForbiddenException('Telemetry timelines are private');
    }
    return this.analyticsService.getUserActivity(
      userId,
      limit ? parseInt(limit.toString(), 10) : 50,
    );
  }

  private getTimeframeMinutes(timeframe: string): number {
    switch (timeframe) {
      case '7d':
        return 7 * 24 * 60;
      case '90d':
        return 90 * 24 * 60;
      case '30d':
      default:
        return 30 * 24 * 60;
    }
  }
}
