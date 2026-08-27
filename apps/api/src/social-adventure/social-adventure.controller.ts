import { Body, Controller, Delete, Get, Param, Post, Put, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import {
  CompleteHumanSkillAttemptDto,
  CreatePackDto,
  CreateSocialShareDto,
  SocialReactionDto,
  UpdateSocialAdventurePreferencesDto,
} from './dto/social-adventure.dto';
import { SocialAdventureService } from './social-adventure.service';

@ApiTags('social-adventure')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('social-adventure')
export class SocialAdventureController {
  constructor(private readonly socialAdventure: SocialAdventureService) {}

  @Get('me')
  getMine(@Request() req: AuthenticatedRequest) {
    return this.socialAdventure.getMine(req.user.sub);
  }

  @Put('preferences')
  updatePreferences(@Request() req: AuthenticatedRequest, @Body() dto: UpdateSocialAdventurePreferencesDto) {
    return this.socialAdventure.updatePreferences(req.user.sub, dto);
  }

  @Get('leaderboard/global')
  getGlobalLeaderboard(@Request() req: AuthenticatedRequest, @Query('limit') limit?: string) {
    return this.socialAdventure.getGlobalLeaderboard(req.user.sub, Number(limit));
  }

  @Get('packs')
  listPacks(@Request() req: AuthenticatedRequest) {
    return this.socialAdventure.listPacks(req.user.sub);
  }

  @Post('packs')
  createPack(@Request() req: AuthenticatedRequest, @Body() dto: CreatePackDto) {
    return this.socialAdventure.createPack(req.user.sub, dto);
  }

  @Post('packs/:packId/join')
  joinPack(@Request() req: AuthenticatedRequest, @Param('packId') packId: string) {
    return this.socialAdventure.joinPack(req.user.sub, packId);
  }

  @Delete('packs/:packId/membership')
  leavePack(@Request() req: AuthenticatedRequest, @Param('packId') packId: string) {
    return this.socialAdventure.leavePack(req.user.sub, packId);
  }

  @Get('packs/:packId/leaderboard')
  getPackLeaderboard(
    @Request() req: AuthenticatedRequest,
    @Param('packId') packId: string,
    @Query('limit') limit?: string
  ) {
    return this.socialAdventure.getPackLeaderboard(req.user.sub, packId, Number(limit));
  }

  @Get('arcade')
  getArcade(@Request() req: AuthenticatedRequest) {
    return this.socialAdventure.getArcade(req.user.sub);
  }

  @Post('arcade/:challengeKey/attempts')
  startAttempt(@Request() req: AuthenticatedRequest, @Param('challengeKey') challengeKey: string) {
    return this.socialAdventure.startHumanSkillAttempt(req.user.sub, challengeKey);
  }

  @Post('arcade/attempts/:attemptId/complete')
  completeAttempt(
    @Request() req: AuthenticatedRequest,
    @Param('attemptId') attemptId: string,
    @Body() dto: CompleteHumanSkillAttemptDto
  ) {
    return this.socialAdventure.completeHumanSkillAttempt(req.user.sub, attemptId, dto);
  }

  @Get('feed')
  getFeed(@Request() req: AuthenticatedRequest, @Query('take') take?: string) {
    return this.socialAdventure.getFeed(req.user.sub, Number(take));
  }

  @Post('shares')
  createShare(@Request() req: AuthenticatedRequest, @Body() dto: CreateSocialShareDto) {
    return this.socialAdventure.createShare(req.user.sub, dto);
  }

  @Post('shares/:shareId/reactions')
  addReaction(
    @Request() req: AuthenticatedRequest,
    @Param('shareId') shareId: string,
    @Body() dto: SocialReactionDto
  ) {
    return this.socialAdventure.addReaction(req.user.sub, shareId, dto);
  }

  @Delete('shares/:shareId/reactions/:reaction')
  removeReaction(
    @Request() req: AuthenticatedRequest,
    @Param('shareId') shareId: string,
    @Param('reaction') reaction: string
  ) {
    return this.socialAdventure.removeReaction(req.user.sub, shareId, reaction);
  }
}
