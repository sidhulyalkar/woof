import { Controller, Get, Post, Put, Param, Body, Query, UseGuards } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { CompatibilityService } from './compatibility.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@ApiTags('compatibility')
@Controller('compatibility')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class CompatibilityController {
  constructor(private compatibilityService: CompatibilityService) {}

  @Post('calculate')
  @ApiOperation({ summary: 'Calculate compatibility score between two pets' })
  @ApiResponse({ status: 200, description: 'Compatibility score calculated' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async calculateCompatibility(
    @Body('petAId') petAId: string,
    @Body('petBId') petBId: string,
  ) {
    return this.compatibilityService.calculateCompatibility(petAId, petBId);
  }

  @Get('recommendations/:petId')
  @ApiOperation({ summary: 'Get compatibility recommendations for a pet' })
  @ApiResponse({ status: 200, description: 'List of recommended compatible pets' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async getRecommendations(
    @Param('petId') petId: string,
    @Query('limit') limit?: number,
  ) {
    return this.compatibilityService.getRecommendations(
      petId,
      limit ? parseInt(limit.toString()) : 10,
    );
  }

  @Put('edge/status')
  @ApiOperation({ summary: 'Update pet edge status (PROPOSED, CONFIRMED, AVOID)' })
  @ApiResponse({ status: 200, description: 'Edge status updated successfully' })
  @ApiResponse({ status: 400, description: 'Invalid status' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async updateEdgeStatus(
    @Body('petAId') petAId: string,
    @Body('petBId') petBId: string,
    @Body('status') status: string,
  ) {
    return this.compatibilityService.updateEdgeStatus(petAId, petBId, status);
  }

  @Get('edges')
  @ApiOperation({ summary: 'Get all pet edges (paginated)' })
  @ApiResponse({ status: 200, description: 'List of pet edges' })
  async getAllEdges(
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('status') status?: string,
  ) {
    return this.compatibilityService.getAllEdges(skip, take, status);
  }

  @Get('edge/:petAId/:petBId')
  @ApiOperation({ summary: 'Get or create pet edge between two pets' })
  @ApiResponse({ status: 200, description: 'Pet edge retrieved or created' })
  @ApiResponse({ status: 404, description: 'Pet not found' })
  async getOrCreatePetEdge(
    @Param('petAId') petAId: string,
    @Param('petBId') petBId: string,
  ) {
    return this.compatibilityService.getOrCreatePetEdge(petAId, petBId);
  }
}
