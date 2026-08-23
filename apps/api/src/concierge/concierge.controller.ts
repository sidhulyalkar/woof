import { Controller, Get, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ConciergeEnabledGuard } from './concierge-enabled.guard';
import { ConciergeService } from './concierge.service';
import { ConciergeQueryDto } from './dto/concierge.dto';

@ApiTags('concierge')
@ApiBearerAuth()
@Controller('concierge')
@UseGuards(JwtAuthGuard, ConciergeEnabledGuard)
export class ConciergeController {
  constructor(private readonly concierge: ConciergeService) {}

  @Get('today')
  @ApiOperation({ summary: 'Compose an explainable suggestion-only briefing for today' })
  getToday(@Request() req: AuthenticatedRequest, @Query() query: ConciergeQueryDto) {
    return this.concierge.getToday(req.user.sub, query.petId);
  }
}
