import { Body, Controller, Post, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { DailySignalsService } from './daily-signals.service';
import { CreateDailySignalsDto } from './dto/daily-signals.dto';

@ApiTags('intelligence')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('intelligence')
export class IntelligenceController {
  constructor(private readonly dailySignals: DailySignalsService) {}

  @Post('daily-signals')
  @ApiOperation({
    summary: 'Record one private, household-clocked Daily Signals check-in for a pet',
  })
  captureDailySignals(@Request() req: AuthenticatedRequest, @Body() dto: CreateDailySignalsDto) {
    return this.dailySignals.capture(req.user.sub, dto);
  }
}
