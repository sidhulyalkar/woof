import { Controller, Get, Post, Body, Query, Param, UseGuards, Request } from '@nestjs/common';
import { ApiBearerAuth, ApiTags, ApiOperation, ApiResponse, ApiQuery } from '@nestjs/swagger';
import { CoActivityService } from './co-activity.service';
import { TrackLocationDto } from './dto/track-location.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('co-activity')
@Controller('co-activity')
export class CoActivityController {
  constructor(private readonly coActivityService: CoActivityService) {}

  @Post('track')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Track user location' })
  @ApiResponse({ status: 201, description: 'Location tracked successfully' })
  async trackLocation(@Request() req: any, @Body() trackLocationDto: TrackLocationDto) {
    return this.coActivityService.trackLocation(req.user.id, trackLocationDto);
  }

  @Get('me/locations')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get current user location history' })
  @ApiQuery({ name: 'hours', required: false, description: 'Hours to look back', example: 24 })
  @ApiResponse({ status: 200, description: 'Location history retrieved' })
  async getMyLocations(@Request() req: any, @Query('hours') hours?: string) {
    const hoursNum = hours ? parseInt(hours, 10) : 24;
    return this.coActivityService.getUserLocations(req.user.id, hoursNum);
  }

  @Get('overlaps/:userId')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Detect co-activity overlaps with another user' })
  @ApiQuery({ name: 'hours', required: false, description: 'Hours to look back', example: 168 })
  @ApiResponse({ status: 200, description: 'Overlaps detected' })
  async detectOverlaps(
    @Request() req: any,
    @Param('userId') userId: string,
    @Query('hours') hours?: string,
  ) {
    const hoursNum = hours ? parseInt(hours, 10) : 168; // Default 7 days
    return this.coActivityService.detectOverlaps(req.user.id, userId, hoursNum);
  }

  @Get('me/matches')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Find potential co-activity matches' })
  @ApiQuery({ name: 'hours', required: false, description: 'Hours to look back', example: 168 })
  @ApiResponse({ status: 200, description: 'Potential matches found' })
  async findMatches(@Request() req: any, @Query('hours') hours?: string) {
    const hoursNum = hours ? parseInt(hours, 10) : 168; // Default 7 days
    return this.coActivityService.findPotentialMatches(req.user.id, hoursNum);
  }

  @Get('me/stats')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get co-activity statistics for current user' })
  @ApiResponse({ status: 200, description: 'Statistics retrieved' })
  async getMyStats(@Request() req: any) {
    return this.coActivityService.getStats(req.user.id);
  }
}
