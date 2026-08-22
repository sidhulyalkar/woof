import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Post,
  Put,
  Query,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ActivitiesService } from './activities.service';
import { CreateActivityDto, UpdateActivityDto } from './dto/activity.dto';

@ApiTags('activities')
@Controller('activities')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class ActivitiesController {
  constructor(private readonly activitiesService: ActivitiesService) {}

  @Post()
  @ApiOperation({ summary: 'Create one household activity for the authenticated recorder' })
  @ApiResponse({ status: 201, description: 'Activity created successfully' })
  async create(@Request() req: AuthenticatedRequest, @Body() dto: CreateActivityDto) {
    return this.activitiesService.create(req.user.sub, dto);
  }

  @Get()
  @ApiOperation({ summary: 'Get activities visible through the authenticated household context' })
  async findAll(
    @Request() req: AuthenticatedRequest,
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('petId') petId?: string
  ) {
    return this.activitiesService.findAll(req.user.sub, skip, take, petId);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get one household-visible activity' })
  async findOne(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.activitiesService.findById(req.user.sub, id);
  }

  @Put(':id')
  @ApiOperation({ summary: 'Update one activity recorded by the authenticated user' })
  async update(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Body() dto: UpdateActivityDto
  ) {
    return this.activitiesService.update(req.user.sub, id, dto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete one activity recorded by the authenticated user' })
  async delete(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.activitiesService.delete(req.user.sub, id);
  }
}
