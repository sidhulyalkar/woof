import { Body, Controller, Delete, Get, Put, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { UpdatePrivacyPreferencesDto } from './dto/update-privacy-preferences.dto';
import { PrivacyService } from './privacy.service';

@ApiTags('privacy')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('privacy')
export class PrivacyController {
  constructor(private readonly privacyService: PrivacyService) {}

  @Get('preferences')
  @ApiOperation({ summary: 'Get current privacy preferences' })
  getPreferences(@Request() req: AuthenticatedRequest) {
    return this.privacyService.getPreferences(req.user.sub);
  }

  @Put('preferences')
  @ApiOperation({ summary: 'Update privacy preferences with location defaulting off' })
  updatePreferences(
    @Request() req: AuthenticatedRequest,
    @Body() dto: UpdatePrivacyPreferencesDto,
  ) {
    return this.privacyService.updatePreferences(req.user.sub, dto);
  }

  @Get('location-summary')
  @ApiOperation({ summary: 'Inspect retained location metadata without returning coordinates' })
  getLocationSummary(@Request() req: AuthenticatedRequest) {
    return this.privacyService.getLocationSummary(req.user.sub);
  }

  @Delete('location-history')
  @ApiOperation({ summary: 'Delete all retained precise location pings' })
  clearLocationHistory(@Request() req: AuthenticatedRequest) {
    return this.privacyService.clearLocationHistory(req.user.sub);
  }
}
