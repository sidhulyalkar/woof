import {
  BadRequestException,
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Post,
  Query,
  Request,
  UploadedFile,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiBearerAuth, ApiConsumes, ApiOperation, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import {
  AnalyzePetHealthDto,
  FollowUpHealthDto,
  HealthTimelineQueryDto,
} from './dto/health-lens.dto';
import { HealthLensService } from './health-lens.service';

const ALLOWED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

@ApiTags('health-lens')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('health-lens')
export class HealthLensController {
  constructor(private readonly healthLens: HealthLensService) {}

  @Post('analyze')
  @ApiConsumes('multipart/form-data')
  @ApiOperation({
    summary: 'Screen a pet health concern from owner context and an optional transient image',
  })
  @UseInterceptors(
    FileInterceptor('image', {
      limits: { fileSize: 8 * 1024 * 1024 },
    }),
  )
  analyze(
    @Request() req: any,
    @Body() dto: AnalyzePetHealthDto,
    @UploadedFile() image?: Express.Multer.File,
  ) {
    if (image && !ALLOWED_IMAGE_TYPES.has(image.mimetype)) {
      throw new BadRequestException('Health Lens accepts JPEG, PNG, or WebP images only');
    }
    return this.healthLens.analyze(req.user.sub, dto, image);
  }

  @Post('follow-up')
  @ApiOperation({ summary: 'Ask a follow-up question about a saved health assessment' })
  followUp(@Request() req: any, @Body() dto: FollowUpHealthDto) {
    return this.healthLens.followUp(req.user.sub, dto);
  }

  @Get('timeline')
  @ApiOperation({ summary: 'Get the derived health observation timeline for one owned pet' })
  timeline(@Request() req: any, @Query() query: HealthTimelineQueryDto) {
    return this.healthLens.timeline(req.user.sub, query.petId, query.limit ?? 20);
  }

  @Delete('timeline/:entryId')
  @ApiOperation({ summary: 'Delete one derived health timeline entry' })
  deleteTimelineEntry(@Request() req: any, @Param('entryId') entryId: string) {
    return this.healthLens.deleteTimelineEntry(req.user.sub, entryId);
  }
}
