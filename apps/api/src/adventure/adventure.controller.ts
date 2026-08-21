import { Body, Controller, Get, Param, Post, Query, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdventureService } from './adventure.service';
import { CompleteQuestDto, QuestInteractionDto } from './dto/adventure.dto';

@ApiTags('adventure')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('adventure')
export class AdventureController {
  constructor(private readonly adventureService: AdventureService) {}

  @Get('me')
  @ApiOperation({ summary: 'Get the personalized Woof Adventure dashboard and daily quest deck' })
  getMine(@Request() req: any, @Query('petId') petId?: string) {
    return this.adventureService.getDashboard(req.user.sub, petId);
  }

  @Post('quests/:questId/select')
  @ApiOperation({ summary: 'Record that the owner selected a generated quest' })
  selectQuest(
    @Request() req: any,
    @Param('questId') questId: string,
    @Body() dto: QuestInteractionDto,
  ) {
    return this.adventureService.recordInteraction(req.user.sub, dto.petId, questId, 'SELECTED');
  }

  @Post('quests/:questId/dismiss')
  @ApiOperation({ summary: 'Record that a generated quest did not fit today' })
  dismissQuest(
    @Request() req: any,
    @Param('questId') questId: string,
    @Body() dto: QuestInteractionDto,
  ) {
    return this.adventureService.recordInteraction(req.user.sub, dto.petId, questId, 'DISMISSED');
  }

  @Post('quests/:questId/complete')
  @ApiOperation({ summary: 'Close the five-second outcome loop and issue server-calculated Bond XP' })
  completeQuest(
    @Request() req: any,
    @Param('questId') questId: string,
    @Body() dto: CompleteQuestDto,
  ) {
    return this.adventureService.completeQuest(req.user.sub, questId, dto);
  }
}
