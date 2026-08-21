import { Controller, Get, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdventureEnabledGuard } from './adventure-enabled.guard';
import { PackChallengesService } from './pack-challenges.service';

@ApiTags('pack')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard, AdventureEnabledGuard)
@Controller('pack')
export class PackChallengesController {
  constructor(private readonly packChallenges: PackChallengesService) {}

  @Get('challenges')
  @ApiOperation({ summary: 'Get cooperative, non-ranking Pack challenges' })
  getChallenges(@Request() req: AuthenticatedRequest) {
    return this.packChallenges.getChallenges(req.user.sub);
  }
}
