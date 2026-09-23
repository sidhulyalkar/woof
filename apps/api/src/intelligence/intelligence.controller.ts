import { Body, Controller, Get, Post, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { DailySignalsCorrectionService } from './daily-signals-correction.service';
import { DailySignalsService } from './daily-signals.service';
import {
  CorrectDailySignalsDto,
  DailySignalsCurrentQueryDto,
} from './dto/daily-signals-correction.dto';
import { CreateDailySignalsDto } from './dto/daily-signals.dto';

@ApiTags('intelligence')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('intelligence')
export class IntelligenceController {
  constructor(
    private readonly dailySignals: DailySignalsService,
    private readonly dailySignalsCorrection: DailySignalsCorrectionService
  ) {}

  @Post('daily-signals')
  @ApiOperation({
    summary: 'Record one private, household-clocked Daily Signals check-in for a pet',
  })
  captureDailySignals(@Request() req: AuthenticatedRequest, @Body() dto: CreateDailySignalsDto) {
    return this.dailySignals.capture(req.user.sub, dto);
  }

  @Get('daily-signals/current')
  @ApiOperation({
    summary:
      'Read the effective structured Daily Signals state for the current household-local day',
  })
  currentDailySignals(
    @Request() req: AuthenticatedRequest,
    @Query() query: DailySignalsCurrentQueryDto
  ) {
    return this.dailySignalsCorrection.getCurrent(req.user.sub, query);
  }

  @Post('daily-signals/corrections')
  @ApiOperation({
    summary: 'Append an authorized correction to the current Daily Signals state',
  })
  correctDailySignals(@Request() req: AuthenticatedRequest, @Body() dto: CorrectDailySignalsDto) {
    return this.dailySignalsCorrection.correct(req.user.sub, dto);
  }
}
