import { Body, Controller, Get, Param, Post, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CaregiverService } from './caregiver.service';
import { CreateCaregiverObservationDto, IssueCaregiverGrantDto } from './dto/caregiver.dto';

@ApiTags('caregiver')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('caregiver')
export class CaregiverController {
  constructor(private readonly caregiver: CaregiverService) {}

  @Post('grants')
  issueGrant(@Request() req: AuthenticatedRequest, @Body() dto: IssueCaregiverGrantDto) {
    return this.caregiver.issueGrant(req.user.sub, dto);
  }

  @Get('grants/issued')
  listIssued(@Request() req: AuthenticatedRequest) {
    return this.caregiver.listIssued(req.user.sub);
  }

  @Get('grants/received')
  listReceived(@Request() req: AuthenticatedRequest) {
    return this.caregiver.listReceived(req.user.sub);
  }

  @Post('grants/:grantId/accept')
  acceptGrant(@Request() req: AuthenticatedRequest, @Param('grantId') grantId: string) {
    return this.caregiver.acceptGrant(req.user.sub, grantId);
  }

  @Post('grants/:grantId/decline')
  declineGrant(@Request() req: AuthenticatedRequest, @Param('grantId') grantId: string) {
    return this.caregiver.declineGrant(req.user.sub, grantId);
  }

  @Post('grants/:grantId/revoke')
  revokeGrant(@Request() req: AuthenticatedRequest, @Param('grantId') grantId: string) {
    return this.caregiver.revokeGrant(req.user.sub, grantId);
  }

  @Get('pets')
  listCaregiverPets(@Request() req: AuthenticatedRequest) {
    return this.caregiver.listCaregiverPets(req.user.sub);
  }

  @Get('pets/:petId/today')
  getCaregiverToday(@Request() req: AuthenticatedRequest, @Param('petId') petId: string) {
    return this.caregiver.getCaregiverToday(req.user.sub, petId);
  }

  @Post('pets/:petId/observations')
  logObservation(
    @Request() req: AuthenticatedRequest,
    @Param('petId') petId: string,
    @Body() dto: CreateCaregiverObservationDto,
  ) {
    return this.caregiver.logObservation(req.user.sub, petId, dto);
  }
}
