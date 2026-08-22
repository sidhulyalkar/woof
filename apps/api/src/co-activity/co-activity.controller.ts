import { Body, Controller, Get, Param, Post, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiQuery, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CoActivityService } from './co-activity.service';
import { TrackLocationDto } from './dto/track-location.dto';

@ApiTags('co-activity')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('co-activity')
export class CoActivityController {
  constructor(private readonly coActivityService: CoActivityService) {}

  @Post('track')
  @ApiOperation({ summary: 'Store a short-lived location ping after explicit consent' })
  trackLocation(@Request() req: AuthenticatedRequest, @Body() dto: TrackLocationDto) {
    return this.coActivityService.trackLocation(req.user.sub, dto);
  }

  @Get('me/location-summary')
  @ApiOperation({ summary: 'Get retained location metadata without coordinates' })
  getLocationSummary(@Request() req: AuthenticatedRequest) {
    return this.coActivityService.getLocationSummary(req.user.sub);
  }

  @Get('overlaps/:userId')
  @ApiOperation({ summary: 'Get a mutual-opt-in proximity summary without coordinates' })
  @ApiQuery({ name: 'hours', required: false, description: 'Lookback, capped at 24 hours' })
  detectOverlaps(
    @Request() req: AuthenticatedRequest,
    @Param('userId') userId: string,
    @Query('hours') hours?: string,
  ) {
    return this.coActivityService.detectOverlaps(
      req.user.sub,
      userId,
      hours ? parseInt(hours, 10) : 12,
    );
  }

  @Get('me/matches')
  @ApiOperation({ summary: 'Find mutual-opt-in proximity candidates without disclosing coordinates' })
  @ApiQuery({ name: 'hours', required: false, description: 'Lookback, capped at 24 hours' })
  findMatches(@Request() req: AuthenticatedRequest, @Query('hours') hours?: string) {
    return this.coActivityService.findPotentialMatches(
      req.user.sub,
      hours ? parseInt(hours, 10) : 12,
    );
  }

  @Get('me/stats')
  @ApiOperation({ summary: 'Get privacy-aware co-activity statistics' })
  getMyStats(@Request() req: AuthenticatedRequest) {
    return this.coActivityService.getStats(req.user.sub);
  }
}
