import { Body, Controller, Delete, Get, Param, Post, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { BlockUserDto, ReportUserDto } from './dto/trust-safety.dto';
import { TrustSafetyService } from './trust-safety.service';

@ApiTags('trust-safety')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('trust-safety')
export class TrustSafetyController {
  constructor(private readonly trustSafetyService: TrustSafetyService) {}

  @Post('blocks')
  @ApiOperation({ summary: 'Block a member and cancel active coordination' })
  blockUser(@Request() req: AuthenticatedRequest, @Body() dto: BlockUserDto) {
    return this.trustSafetyService.blockUser(req.user.sub, dto);
  }

  @Delete('blocks/:userId')
  @ApiOperation({ summary: 'Remove a block' })
  unblockUser(@Request() req: AuthenticatedRequest, @Param('userId') blockedUserId: string) {
    return this.trustSafetyService.unblockUser(req.user.sub, blockedUserId);
  }

  @Get('blocks')
  @ApiOperation({ summary: 'List blocked members' })
  getBlockedUsers(@Request() req: AuthenticatedRequest) {
    return this.trustSafetyService.getBlockedUsers(req.user.sub);
  }

  @Post('reports')
  @ApiOperation({ summary: 'Submit a safety or conduct report' })
  reportUser(@Request() req: AuthenticatedRequest, @Body() dto: ReportUserDto) {
    return this.trustSafetyService.reportUser(req.user.sub, dto);
  }

  @Get('reports')
  @ApiOperation({ summary: 'Get status of reports submitted by the current member' })
  getMyReports(@Request() req: AuthenticatedRequest) {
    return this.trustSafetyService.getMyReports(req.user.sub);
  }
}
