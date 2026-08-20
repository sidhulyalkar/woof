import {
  Body,
  Controller,
  Get,
  Param,
  Post,
  Query,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { RecommendationFeedbackDto } from './dto/recommendation-feedback.dto';
import { InsightsService } from './insights.service';

@ApiTags('insights')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('insights')
export class InsightsController {
  constructor(private readonly insightsService: InsightsService) {}

  @Get('me')
  @ApiOperation({
    summary: 'Get personalized daily recommendations and relationship-learning signals',
  })
  async getMine(@Request() req: any, @Query('petId') petId?: string) {
    return this.insightsService.getForUser(req.user.sub, petId);
  }

  @Post('pets/:petId/recommendation-feedback')
  @ApiOperation({
    summary: 'Record feedback used to adapt future recommendation ranking',
  })
  async recordFeedback(
    @Request() req: any,
    @Param('petId') petId: string,
    @Body() dto: RecommendationFeedbackDto,
  ) {
    return this.insightsService.recordFeedback(req.user.sub, petId, dto);
  }
}
