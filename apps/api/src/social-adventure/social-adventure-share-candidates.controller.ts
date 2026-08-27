import { Controller, Get, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { SocialAdventureShareCandidatesService } from './social-adventure-share-candidates.service';

@ApiTags('social-adventure')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('social-adventure/share-candidates')
export class SocialAdventureShareCandidatesController {
  constructor(private readonly candidates: SocialAdventureShareCandidatesService) {}

  @Get()
  list(@Request() req: AuthenticatedRequest) {
    return this.candidates.list(req.user.sub);
  }
}
