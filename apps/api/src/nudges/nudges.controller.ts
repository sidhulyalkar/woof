import {
  Controller,
  Get,
  Param,
  Patch,
  Post,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { NudgesService } from './nudges.service';

@ApiTags('nudges')
@ApiBearerAuth()
@Controller('nudges')
@UseGuards(JwtAuthGuard)
export class NudgesController {
  constructor(private readonly nudgesService: NudgesService) {}

  @Get()
  @ApiOperation({ summary: 'Get active nudges for the authenticated user' })
  async getUserNudges(@Request() req: AuthenticatedRequest) {
    return this.nudgesService.getUserNudges(req.user.sub);
  }

  @Patch(':id/dismiss')
  @ApiOperation({ summary: 'Dismiss an owned nudge and record negative feedback' })
  async dismissNudge(@Param('id') id: string, @Request() req: AuthenticatedRequest) {
    return this.nudgesService.dismissNudge(id, req.user.sub);
  }

  @Patch(':id/accept')
  @ApiOperation({ summary: 'Accept an owned nudge and record positive feedback' })
  async acceptNudge(@Param('id') id: string, @Request() req: AuthenticatedRequest) {
    return this.nudgesService.acceptNudge(id, req.user.sub);
  }

  @Post('check/chat/:conversationId')
  @ApiOperation({ summary: 'Create an in-app suggestion from an owned conversation when useful' })
  async checkChatActivity(
    @Param('conversationId') conversationId: string,
    @Request() req: AuthenticatedRequest,
  ) {
    return this.nudgesService.checkChatActivityNudges(conversationId, req.user.sub);
  }
}
