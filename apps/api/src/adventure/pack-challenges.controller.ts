import { Controller, Get, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { PackChallengesService } from './pack-challenges.service';

@ApiTags('pack')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('pack')
export class PackChallengesController {
  constructor(private readonly packChallenges: PackChallengesService) {}

  @Get('challenges')
  @ApiOperation({ summary: 'Get cooperative, non-ranking Pack challenges' })
  getChallenges(@Request() req: any) {
    return this.packChallenges.getChallenges(req.user.sub);
  }
}
