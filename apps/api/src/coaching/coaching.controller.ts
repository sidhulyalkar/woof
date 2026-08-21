import {
  Body,
  Controller,
  Get,
  Param,
  Patch,
  Post,
  Query,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CareEventsService } from '../care-events/care-events.service';
import { CoachingService } from './coaching.service';
import {
  CreateTrainingPlanDto,
  RecordTrainingSessionDto,
  UpdateTrainingPlanStatusDto,
} from './dto/coaching.dto';

@ApiTags('coaching')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('coaching')
export class CoachingController {
  constructor(
    private readonly coachingService: CoachingService,
    private readonly careEvents: CareEventsService,
  ) {}

  @Get('me')
  @ApiOperation({ summary: 'Get the current reward-based coaching plan and practice context' })
  getMine(@Request() req: any, @Query('petId') petId?: string) {
    return this.coachingService.getDashboard(req.user.sub, petId);
  }

  @Post('plans')
  @ApiOperation({ summary: 'Start one focused coaching plan for an owned pet' })
  createPlan(@Request() req: any, @Body() dto: CreateTrainingPlanDto) {
    return this.coachingService.createPlan(req.user.sub, dto);
  }

  @Patch('plans/:planId/status')
  @ApiOperation({ summary: 'Pause or resume an owned coaching plan' })
  updatePlanStatus(
    @Request() req: any,
    @Param('planId') planId: string,
    @Body() dto: UpdateTrainingPlanStatusDto,
  ) {
    return this.coachingService.setPlanStatus(req.user.sub, planId, dto);
  }

  @Post('plans/:planId/sessions')
  @ApiOperation({ summary: 'Record an observable practice session and adapt the next difficulty' })
  async recordSession(
    @Request() req: any,
    @Param('planId') planId: string,
    @Body() dto: RecordTrainingSessionDto,
  ) {
    const result = await this.coachingService.recordSession(req.user.sub, planId, dto);
    const petId = result.plan?.petId;

    if (petId) {
      const concernSignals = dto.stressSignals ?? [];
      const listenedAndStopped = Boolean(dto.stoppedEarly && concernSignals.length > 0);
      await this.careEvents.record({
        userId: req.user.sub,
        petId,
        eventType: listenedAndStopped ? 'SAFE_OPT_OUT' : 'TRAINING_SESSION',
        pathway: listenedAndStopped ? 'BOND' : 'LEARN',
        source: 'WOOF_COACH',
        evidenceType: 'COACH',
        evidenceConfidence: 0.86,
        dedupeKey: `coach:${result.activityId}`,
        safetyEligible: listenedAndStopped || concernSignals.length === 0,
        context: {
          planId,
          activityId: result.activityId,
          attempts: dto.attempts,
          successes: dto.successes,
          durationSeconds: dto.durationSeconds,
        },
        outcome: {
          stressSignals: concernSignals,
          stoppedEarly: dto.stoppedEarly ?? false,
          safeOptOut: listenedAndStopped,
        },
      });
    }

    return result;
  }
}
