import { Controller, Get, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { MLService } from './ml.service';

@ApiTags('ml')
@ApiBearerAuth()
@Controller('ml')
@UseGuards(JwtAuthGuard)
export class MLController {
  constructor(private readonly mlService: MLService) {}

  @Get('status')
  @ApiOperation({
    summary: 'Get compatibility model integration status',
    description:
      'Returns configuration state only. Product clients request compatibility through the compatibility API so learned and deterministic scorers share one stable contract.',
  })
  getStatus() {
    return this.mlService.getStatus();
  }
}
