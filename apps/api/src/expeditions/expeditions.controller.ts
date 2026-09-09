import { Controller, Get, Param, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ExpeditionsService } from './expeditions.service';

@ApiTags('expeditions')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('expeditions')
export class ExpeditionsController {
  constructor(private readonly expeditions: ExpeditionsService) {}

  @Get('global')
  @ApiOperation({ summary: 'Get receipt-backed Global Expedition progress' })
  getGlobal(@Request() req: AuthenticatedRequest) {
    return this.expeditions.getGlobal(req.user.sub);
  }

  @Get('packs/:packId')
  @ApiOperation({ summary: 'Get receipt-backed Pack Expedition progress for an active member' })
  getPack(@Request() req: AuthenticatedRequest, @Param('packId') packId: string) {
    return this.expeditions.getPack(req.user.sub, packId);
  }
}
