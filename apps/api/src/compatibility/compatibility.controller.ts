import {
  Body,
  Controller,
  Get,
  Param,
  Post,
  Put,
  Query,
  Request,
  UseGuards,
} from '@nestjs/common';
import {
  ApiBearerAuth,
  ApiOperation,
  ApiResponse,
  ApiTags,
} from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CompatibilityService } from './compatibility.service';

@ApiTags('compatibility')
@Controller('compatibility')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class CompatibilityController {
  constructor(private readonly compatibilityService: CompatibilityService) {}

  @Post('calculate')
  @ApiOperation({ summary: 'Calculate compatibility for an owned pet relationship' })
  @ApiResponse({ status: 200, description: 'Compatibility score calculated' })
  async calculateCompatibility(
    @Request() req: AuthenticatedRequest,
    @Body('petAId') petAId: string,
    @Body('petBId') petBId: string,
  ) {
    return this.compatibilityService.calculateCompatibility(req.user.sub, petAId, petBId);
  }

  @Get('recommendations/:petId')
  @ApiOperation({ summary: 'Get ranked compatibility recommendations for an owned pet' })
  async getRecommendations(
    @Request() req: AuthenticatedRequest,
    @Param('petId') petId: string,
    @Query('limit') limit?: number,
  ) {
    return this.compatibilityService.getRecommendations(
      req.user.sub,
      petId,
      limit ? parseInt(limit.toString(), 10) : 10,
    );
  }

  @Put('edge/status')
  @ApiOperation({ summary: 'Update the status of a relationship involving an owned pet' })
  async updateEdgeStatus(
    @Request() req: AuthenticatedRequest,
    @Body('petAId') petAId: string,
    @Body('petBId') petBId: string,
    @Body('status') status: string,
  ) {
    return this.compatibilityService.updateEdgeStatus(req.user.sub, petAId, petBId, status);
  }

  @Get('edges')
  @ApiOperation({ summary: 'Get relationship edges involving the authenticated user’s pets' })
  async getAllEdges(
    @Request() req: AuthenticatedRequest,
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('status') status?: string,
  ) {
    return this.compatibilityService.getAllEdges(req.user.sub, skip, take, status);
  }

  @Get('edge/:petAId/:petBId')
  @ApiOperation({ summary: 'Get or create an owned pet relationship edge' })
  async getOrCreatePetEdge(
    @Request() req: AuthenticatedRequest,
    @Param('petAId') petAId: string,
    @Param('petBId') petBId: string,
  ) {
    return this.compatibilityService.getOrCreatePetEdgeForActor(req.user.sub, petAId, petBId);
  }
}
