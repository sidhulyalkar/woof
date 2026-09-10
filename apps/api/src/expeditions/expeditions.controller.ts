import { Controller, Get, Param, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ExpeditionJournalService } from './expedition-journal.service';
import { ExpeditionsService } from './expeditions.service';

@ApiTags('expeditions')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('expeditions')
export class ExpeditionsController {
  constructor(
    private readonly expeditions: ExpeditionsService,
    private readonly journal: ExpeditionJournalService
  ) {}

  @Get('global')
  @ApiOperation({ summary: 'Get receipt-backed Global Expedition progress' })
  getGlobal(@Request() req: AuthenticatedRequest) {
    return this.expeditions.getGlobal(req.user.sub);
  }

  @Get('journal')
  @ApiOperation({ summary: 'Get the signed-in human’s receipt-backed Expedition field journal' })
  getJournal(@Request() req: AuthenticatedRequest) {
    return this.journal.getMine(req.user.sub);
  }

  @Get('packs/:packId')
  @ApiOperation({ summary: 'Get receipt-backed Pack Expedition progress for an active member' })
  getPack(@Request() req: AuthenticatedRequest, @Param('packId') packId: string) {
    return this.expeditions.getPack(req.user.sub, packId);
  }
}
