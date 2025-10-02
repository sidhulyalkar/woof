import { Controller, Get, Post, Put, Delete, Param, Body, Query, UseGuards } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { ActivitiesService } from './activities.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { Prisma } from '@woof/database';

@ApiTags('activities')
@Controller('activities')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class ActivitiesController {
  constructor(private activitiesService: ActivitiesService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new activity' })
  @ApiResponse({ status: 201, description: 'Activity created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async create(@Body() createActivityDto: Prisma.ActivityCreateInput) {
    return this.activitiesService.create(createActivityDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all activities (paginated)' })
  @ApiResponse({ status: 200, description: 'List of activities' })
  async findAll(
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('userId') userId?: string,
    @Query('petId') petId?: string,
  ) {
    return this.activitiesService.findAll(skip, take, userId, petId);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get activity by ID' })
  @ApiResponse({ status: 200, description: 'Activity found' })
  @ApiResponse({ status: 404, description: 'Activity not found' })
  async findOne(@Param('id') id: string) {
    return this.activitiesService.findById(id);
  }

  @Put(':id')
  @ApiOperation({ summary: 'Update activity by ID' })
  @ApiResponse({ status: 200, description: 'Activity updated successfully' })
  @ApiResponse({ status: 404, description: 'Activity not found' })
  async update(
    @Param('id') id: string,
    @Body() updateActivityDto: Prisma.ActivityUpdateInput,
  ) {
    return this.activitiesService.update(id, updateActivityDto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete activity by ID' })
  @ApiResponse({ status: 200, description: 'Activity deleted successfully' })
  @ApiResponse({ status: 404, description: 'Activity not found' })
  async delete(@Param('id') id: string) {
    return this.activitiesService.delete(id);
  }
}
