import { Controller, Get, Query } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiQuery } from '@nestjs/swagger';
import { AnalyticsService } from './analytics.service';

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
