import { Body, Controller, Get, Put, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { StoryQueryDto, UpdateStoryCurationDto } from './dto/story.dto';
import { StoryEnabledGuard } from './story-enabled.guard';
import { StoryService } from './story.service';

@ApiTags('story')
@ApiBearerAuth()
@Controller('story')
@UseGuards(JwtAuthGuard, StoryEnabledGuard)
export class StoryController {
  constructor(private readonly story: StoryService) {}

  @Get()
  @ApiOperation({ summary: 'Read the unified dogOS life story without duplicating source truth' })
  getStory(@Request() req: AuthenticatedRequest, @Query() query: StoryQueryDto) {
    return this.story.getStory(req.user.sub, query);
  }

  @Put('curation')
  @ApiOperation({ summary: 'Save, annotate, or clear one Story source reference' })
  updateCuration(@Request() req: AuthenticatedRequest, @Body() dto: UpdateStoryCurationDto) {
    return this.story.updateCuration(req.user.sub, dto);
  }
}
