import { Body, Controller, Get, Post, Request, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { SaveQuizResponseDto } from './dto/save-quiz-response.dto';
import { QuizService } from './quiz.service';

@ApiTags('quiz')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('quiz')
export class QuizController {
  constructor(private readonly quizService: QuizService) {}

  @Post('responses')
  @ApiOperation({ summary: 'Save the authenticated user’s onboarding/matching preferences' })
  @ApiResponse({ status: 201, description: 'Preference responses persisted' })
  async save(@Request() req: AuthenticatedRequest, @Body() dto: SaveQuizResponseDto) {
    return this.quizService.save(req.user.sub, dto);
  }

  @Get('responses/latest')
  @ApiOperation({ summary: 'Get the authenticated user’s most recent preference session' })
  async latest(@Request() req: AuthenticatedRequest) {
    return this.quizService.latest(req.user.sub);
  }
}
