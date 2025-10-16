import {
  Controller,
  Get,
  Post,
  Param,
  Body,
  UseGuards,
  Request,
  Patch,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiBearerAuth } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { NudgesService } from './nudges.service';
import { CreateNudgeDto } from './dto/create-nudge.dto';
import { Request as ExpressRequest } from 'express';

@ApiTags('nudges')
@ApiBearerAuth()
@Controller('nudges')
@UseGuards(JwtAuthGuard)
export class NudgesController {
  constructor(private readonly nudgesService: NudgesService) {}

  @Get()
  @ApiOperation({ summary: 'Get active nudges for current user' })
  async getUserNudges(@Request() req: ExpressRequest & { user: any }) {
    return this.nudgesService.getUserNudges(req.user.id);
  }

  @Post()
  @ApiOperation({ summary: 'Create a manual nudge (admin/testing)' })
  async createNudge(@Body() createNudgeDto: CreateNudgeDto) {
    return this.nudgesService.createNudge(createNudgeDto);
  }

  @Patch(':id/dismiss')
  @ApiOperation({ summary: 'Dismiss a nudge' })
  async dismissNudge(@Param('id') id: string, @Request() req: ExpressRequest & { user: any }) {
    return this.nudgesService.dismissNudge(id, req.user.id);
  }

  @Patch(':id/accept')
  @ApiOperation({ summary: 'Accept a nudge' })
  async acceptNudge(@Param('id') id: string, @Request() req: ExpressRequest & { user: any }) {
    return this.nudgesService.acceptNudge(id, req.user.id);
  }

  @Post('check/chat/:conversationId')
  @ApiOperation({ summary: 'Manually trigger chat activity check' })
  async checkChatActivity(@Param('conversationId') conversationId: string) {
    await this.nudgesService.checkChatActivityNudges(conversationId);
    return { message: 'Chat activity check triggered' };
  }
}
