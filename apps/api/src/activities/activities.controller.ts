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
import {
  ApiBearerAuth,
  ApiOperation,
  ApiResponse,
  ApiTags,
} from '@nestjs/swagger';
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
  @ApiOperation({ summary: 'Create an activity for the authenticated owner' })
  @ApiResponse({ status: 201, description: 'Activity created successfully' })
  async create(@Request() req: any, @Body() dto: CreateActivityDto) {
    return this.activitiesService.create(req.user.sub, dto);
  }

  @Get()
  @ApiOperation({ summary: 'Get the authenticated owner’s activities' })
  async findAll(
    @Request() req: any,
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('petId') petId?: string,
  ) {
    return this.activitiesService.findAll(req.user.sub, skip, take, petId);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get one owned activity' })
  async findOne(@Request() req: any, @Param('id') id: string) {
    return this.activitiesService.findById(req.user.sub, id);
  }

  @Put(':id')
  @ApiOperation({ summary: 'Update one owned activity' })
  async update(
    @Request() req: any,
    @Param('id') id: string,
    @Body() dto: UpdateActivityDto,
  ) {
    return this.activitiesService.update(req.user.sub, id, dto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete one owned activity' })
  async delete(@Request() req: any, @Param('id') id: string) {
    return this.activitiesService.delete(req.user.sub, id);
  }
}
