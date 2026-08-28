import { Body, Controller, Get, Put, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { UpdateCompanionModeDto, UpdateReadinessReflectionDto } from './dto/companion.dto';
import { CompanionService } from './companion.service';

@ApiTags('companion')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('companion')
export class CompanionController {
  constructor(private readonly companion: CompanionService) {}

  @Get('state')
  getState(@Request() req: AuthenticatedRequest) {
    return this.companion.getState(req.user.sub);
  }

  @Put('mode')
  updateMode(@Request() req: AuthenticatedRequest, @Body() dto: UpdateCompanionModeDto) {
    return this.companion.updateMode(req.user.sub, dto.mode);
  }

  @Get('readiness')
  getReadiness(@Request() req: AuthenticatedRequest) {
    return this.companion.getReadiness(req.user.sub);
  }

  @Put('readiness')
  updateReadiness(
    @Request() req: AuthenticatedRequest,
    @Body() dto: UpdateReadinessReflectionDto
  ) {
    return this.companion.updateReadiness(req.user.sub, dto);
  }
}
