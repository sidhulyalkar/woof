import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  Query,
  UseGuards,
  Request,
} from '@nestjs/common';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { GoalsService } from './goals.service';
import { CreateGoalDto, UpdateGoalDto } from './dto';

@Controller('goals')
@UseGuards(JwtAuthGuard)
export class GoalsController {
  constructor(private readonly goalsService: GoalsService) {}

  @Post()
  create(@Request() req: AuthenticatedRequest, @Body() createGoalDto: CreateGoalDto) {
    return this.goalsService.create(req.user.sub, createGoalDto);
  }

  @Get()
  findAll(
    @Request() req: AuthenticatedRequest,
    @Query('petId') petId?: string,
    @Query('status') status?: string
  ) {
    return this.goalsService.findAll(req.user.sub, petId, status);
  }

  @Get('statistics')
  getStatistics(@Request() req: AuthenticatedRequest) {
    return this.goalsService.getStatistics(req.user.sub);
  }

  @Get(':id')
  findOne(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.goalsService.findOne(req.user.sub, id);
  }

  @Patch(':id')
  update(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Body() updateGoalDto: UpdateGoalDto
  ) {
    return this.goalsService.update(req.user.sub, id, updateGoalDto);
  }

  @Patch(':id/progress')
  updateProgress(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Body('value') value: number
  ) {
    return this.goalsService.updateProgress(req.user.sub, id, value);
  }

  @Delete(':id')
  remove(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.goalsService.remove(req.user.sub, id);
  }
}
