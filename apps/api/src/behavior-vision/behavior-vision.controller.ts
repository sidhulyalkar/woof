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
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { BehaviorShadowService } from './behavior-shadow.service';
import { BehaviorVisionService } from './behavior-vision.service';
import {
  AnalyzeBehaviorMediaDto,
  BehaviorObservationFeedbackDto,
  BehaviorTimelineQueryDto,
} from './dto/behavior-vision.dto';

const ALLOWED_MEDIA_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'video/mp4',
  'video/webm',
  'video/quicktime',
]);

@ApiTags('behavior-vision')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('behavior-vision')
export class BehaviorVisionController {
  constructor(
    private readonly behaviorVision: BehaviorVisionService,
    private readonly behaviorShadow: BehaviorShadowService
  ) {}

  @Post('analyze')
  @ApiConsumes('multipart/form-data')
  @ApiOperation({
    summary: 'Analyze a transient pet image/video and update the individual behavior model',
  })
  @UseInterceptors(
    FileInterceptor('media', {
      limits: { fileSize: 50 * 1024 * 1024 },
    })
  )
  analyze(
    @Request() req: AuthenticatedRequest,
    @Body() dto: AnalyzeBehaviorMediaDto,
    @UploadedFile() media?: Express.Multer.File
  ) {
    if (!media) throw new BadRequestException('Behavior Vision requires an image or video');
    if (!ALLOWED_MEDIA_TYPES.has(media.mimetype)) {
      throw new BadRequestException(
        'Behavior Vision accepts JPEG, PNG, WebP, MP4, WebM, or QuickTime media only'
      );
    }
    return this.behaviorVision.analyze(req.user.sub, dto, media);
  }

  @Get('profile')
  @ApiOperation({ summary: 'Get the individualized behavior profile for one owned pet' })
  profile(@Request() req: AuthenticatedRequest, @Query() query: BehaviorTimelineQueryDto) {
    return this.behaviorVision.profile(req.user.sub, query.petId);
  }

  @Get('timeline')
  @ApiOperation({ summary: 'Get recent derived behavior observations for one owned pet' })
  timeline(@Request() req: AuthenticatedRequest, @Query() query: BehaviorTimelineQueryDto) {
    return this.behaviorVision.timeline(req.user.sub, query.petId, query.limit ?? 30);
  }

  @Get('shadow')
  @ApiOperation({
    summary:
      'Inspect Behavior Moments evidence and promotion-readiness metrics with zero authority',
  })
  shadow(@Request() req: AuthenticatedRequest, @Query() query: BehaviorTimelineQueryDto) {
    return this.behaviorShadow.snapshot(req.user.sub, query.petId);
  }

  @Post('feedback')
  @ApiOperation({ summary: 'Correct or confirm an automated behavior observation' })
  feedback(@Request() req: AuthenticatedRequest, @Body() dto: BehaviorObservationFeedbackDto) {
    return this.behaviorVision.recordFeedback(req.user.sub, dto);
  }

  @Delete('observations/:observationId')
  @ApiOperation({ summary: 'Delete one derived behavior observation and its feedback' })
  deleteObservation(
    @Request() req: AuthenticatedRequest,
    @Param('observationId') observationId: string
  ) {
    return this.behaviorVision.deleteObservation(req.user.sub, observationId);
  }
}
