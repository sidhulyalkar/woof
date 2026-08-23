import { Body, Controller, Delete, Get, Param, Post, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { IngestTrackerObservationDto } from '../autopilot/dto/autopilot.dto';
import { ConnectorsEnabledGuard } from './connectors-enabled.guard';
import { ConnectorsService } from './connectors.service';

@ApiTags('connectors')
@ApiBearerAuth()
@Controller('connectors')
@UseGuards(JwtAuthGuard, ConnectorsEnabledGuard)
export class ConnectorsController {
  constructor(private readonly connectors: ConnectorsService) {}

  @Get()
  @ApiOperation({ summary: 'List dogOS connector capabilities and verified connection state' })
  getDashboard(@Request() req: AuthenticatedRequest) {
    return this.connectors.getDashboard(req.user.sub);
  }

  @Post(':provider/oauth/start')
  @ApiOperation({ summary: 'Start OAuth only when an official provider transport is configured' })
  startOAuth(@Param('provider') provider: string) {
    return this.connectors.startOAuth(provider);
  }

  @Delete(':provider')
  @ApiOperation({ summary: 'Remove locally stored connector credentials' })
  disconnect(@Request() req: AuthenticatedRequest, @Param('provider') provider: string) {
    return this.connectors.disconnect(req.user.sub, provider);
  }

  @Post(':provider/import/wearable')
  @ApiOperation({
    summary: 'Import a verified wearable summary through the existing zero-reward Autopilot path',
  })
  importWearable(
    @Request() req: AuthenticatedRequest,
    @Param('provider') provider: string,
    @Body() dto: IngestTrackerObservationDto,
  ) {
    return this.connectors.importWearableObservation(req.user.sub, provider, dto);
  }
}
