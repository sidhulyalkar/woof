import { Controller, Get, Request, UseGuards } from '@nestjs/common';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ABTestService } from './ab-test.service';

/**
 * Legacy experiment-assignment surface.
 *
 * Woof no longer accepts client-authored prediction/outcome events or exposes
 * aggregate experiment administration through the member API. Compatibility
 * provenance and post-meetup outcomes are recorded by the server-side product
 * flow instead. This endpoint remains only for deterministic assignment while
 * the older experiment service is retired.
 */
@Controller('ab-test')
@UseGuards(JwtAuthGuard)
export class ABTestController {
  constructor(private readonly abTestService: ABTestService) {}

  @Get('variant')
  getVariant(@Request() req: AuthenticatedRequest) {
    return {
      variant: this.abTestService.assignVariant(req.user.sub),
      authority: 'experimental-only',
    };
  }
}
