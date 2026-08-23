import { Body, Controller, Delete, Get, Param, Put, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { DiscoveryService } from './discovery.service';
import { UpdateDiscoveryLocationDto } from './dto/discovery-location.dto';

@ApiTags('discovery')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('discovery')
export class DiscoveryController {
  constructor(private readonly discoveryService: DiscoveryService) {}

  @Put('location')
  @ApiOperation({ summary: 'Opt into nearby discovery using a coarse server-derived location cell' })
  updateLocation(
    @Request() req: AuthenticatedRequest,
    @Body() dto: UpdateDiscoveryLocationDto,
  ) {
    return this.discoveryService.updateLocation(req.user.sub, dto.latitude, dto.longitude);
  }

  @Delete('location')
  @ApiOperation({ summary: 'Disable nearby discovery for the signed-in member' })
  disableLocation(@Request() req: AuthenticatedRequest) {
    return this.discoveryService.disableLocation(req.user.sub);
  }

  @Get('location')
  @ApiOperation({ summary: 'Get discovery consent and freshness status without coordinates' })
  getLocationStatus(@Request() req: AuthenticatedRequest) {
    return this.discoveryService.getStatus(req.user.sub);
  }

  @Get('nearby/:petId')
  @ApiOperation({ summary: 'Find coarse-distance public candidates without returning coordinates' })
  getNearby(
    @Request() req: AuthenticatedRequest,
    @Param('petId') petId: string,
    @Query('radiusKm') radiusKm?: string,
    @Query('limit') limit?: string,
  ) {
    return this.discoveryService.getNearbyCandidates(
      req.user.sub,
      petId,
      radiusKm ? Number(radiusKm) : 5,
      limit ? Number(limit) : 20,
    );
  }
}
