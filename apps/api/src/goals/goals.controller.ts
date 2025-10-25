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
import { GoalsService } from './goals.service';
import { CreateGoalDto, UpdateGoalDto } from './dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Controller('goals')
@UseGuards(JwtAuthGuard)
export class GoalsController {
  constructor(private readonly goalsService: GoalsService) {}

  @Post()
  create(@Request() req, @Body() createGoalDto: CreateGoalDto) {
    return this.goalsService.create(req.user.userId, createGoalDto);
  }

  @Get()
  findAll(
    @Request() req,
    @Query('petId') petId?: string,
    @Query('status') status?: string,
  ) {
    return this.goalsService.findAll(req.user.userId, petId, status);
  }

  @Get('statistics')
  getStatistics(@Request() req) {
    return this.goalsService.getStatistics(req.user.userId);
  }

  @Get(':id')
  findOne(@Request() req, @Param('id') id: string) {
    return this.goalsService.findOne(req.user.userId, id);
  }

  @Patch(':id')
  update(
    @Request() req,
    @Param('id') id: string,
    @Body() updateGoalDto: UpdateGoalDto,
  ) {
    return this.goalsService.update(req.user.userId, id, updateGoalDto);
  }

  @Patch(':id/progress')
  updateProgress(
    @Request() req,
    @Param('id') id: string,
    @Body('value') value: number,
  ) {
    return this.goalsService.updateProgress(req.user.userId, id, value);
  }

  @Delete(':id')
  remove(@Request() req, @Param('id') id: string) {
    return this.goalsService.remove(req.user.userId, id);
  }
}
