import { Body, Controller, Delete, Get, Param, Post, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AutopilotEnabledGuard } from './autopilot-enabled.guard';
import { AutopilotService } from './autopilot.service';
import { CreateCareReminderDto, IngestTrackerObservationDto } from './dto/autopilot.dto';

@ApiTags('autopilot')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard, AutopilotEnabledGuard)
@Controller('autopilot')
export class AutopilotController {
  constructor(private readonly autopilot: AutopilotService) {}

  @Get()
  @ApiOperation({ summary: 'Get reminders, non-diagnostic signals, and provider capabilities' })
  getDashboard(@Request() req: AuthenticatedRequest) {
    return this.autopilot.getDashboard(req.user.sub);
  }

  @Post('observations/:provider')
  @ApiOperation({
    summary: 'Normalize a connected tracker summary into a private zero-reward observation',
  })
  ingestObservation(
    @Request() req: AuthenticatedRequest,
    @Param('provider') provider: string,
    @Body() dto: IngestTrackerObservationDto,
  ) {
    return this.autopilot.ingestProviderObservation(req.user.sub, provider, dto);
  }

  @Post('reminders')
  @ApiOperation({ summary: 'Schedule a dogOS care reminder' })
  createReminder(@Request() req: AuthenticatedRequest, @Body() dto: CreateCareReminderDto) {
    return this.autopilot.createReminder(req.user.sub, dto);
  }

  @Delete('reminders/:id')
  @ApiOperation({ summary: 'Cancel a scheduled care reminder' })
  cancelReminder(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.autopilot.cancelReminder(req.user.sub, id);
  }

  @Post('signals/:id/acknowledge')
  @ApiOperation({ summary: 'Acknowledge a non-diagnostic Autopilot signal' })
  acknowledgeSignal(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.autopilot.acknowledgeSignal(req.user.sub, id);
  }
}
