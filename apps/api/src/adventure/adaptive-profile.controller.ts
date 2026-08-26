import { Body, Controller, Get, Param, Post, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdaptiveProfileService } from './adaptive-profile.service';
import { AdventureEnabledGuard } from './adventure-enabled.guard';
import {
  CorrectAdaptiveProfileDto,
  RecordProfileQuestionResponseDto,
} from './dto/adaptive-profile.dto';

@ApiTags('adventure-profile')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard, AdventureEnabledGuard)
@Controller('adventure/profile/:householdId/:petId')
export class AdaptiveProfileController {
  constructor(private readonly adaptiveProfile: AdaptiveProfileService) {}

  @Get()
  @ApiOperation({ summary: 'Get the authorized household/pet Adaptive Adventure profile state' })
  getState(
    @Request() req: AuthenticatedRequest,
    @Param('householdId') householdId: string,
    @Param('petId') petId: string
  ) {
    return this.adaptiveProfile.getState(req.user.sub, householdId, petId);
  }

  @Post('questions/respond')
  @ApiOperation({
    summary: 'Record a replay-safe progressive-profile question response without awarding XP',
  })
  recordQuestionResponse(
    @Request() req: AuthenticatedRequest,
    @Param('householdId') householdId: string,
    @Param('petId') petId: string,
    @Body() dto: RecordProfileQuestionResponseDto
  ) {
    return this.adaptiveProfile.recordQuestionResponse(req.user.sub, householdId, petId, dto);
  }

  @Post('correct')
  @ApiOperation({ summary: 'Append an authoritative owner correction to the pair profile' })
  correct(
    @Request() req: AuthenticatedRequest,
    @Param('householdId') householdId: string,
    @Param('petId') petId: string,
    @Body() dto: CorrectAdaptiveProfileDto
  ) {
    return this.adaptiveProfile.correct(req.user.sub, householdId, petId, dto);
  }
}
