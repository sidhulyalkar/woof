import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Post,
  Put,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CreateMeetupProposalDto } from './dto/create-meetup-proposal.dto';
import {
  CompleteMeetupDto,
  UpdateMeetupProposalDto,
} from './dto/update-meetup-proposal.dto';
import { MeetupProposalsService } from './meetup-proposals.service';

@ApiTags('meetup-proposals')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('meetup-proposals')
export class MeetupProposalsController {
  constructor(private readonly meetupProposalsService: MeetupProposalsService) {}

  @Post()
  @ApiOperation({ summary: 'Propose a privacy-minimized meetup after conversation' })
  create(@Request() req: AuthenticatedRequest, @Body() dto: CreateMeetupProposalDto) {
    return this.meetupProposalsService.create(req.user.sub, dto);
  }

  @Get()
  @ApiOperation({ summary: 'Get sent and received meetup proposals' })
  findAll(@Request() req: AuthenticatedRequest) {
    return this.meetupProposalsService.findAllForUser(req.user.sub);
  }

  @Get('stats')
  @ApiOperation({ summary: 'Get the current member meetup statistics' })
  getStats(@Request() req: AuthenticatedRequest) {
    return this.meetupProposalsService.getStats(req.user.sub);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a meetup proposal only when the member is a participant' })
  findOne(@Param('id') id: string, @Request() req: AuthenticatedRequest) {
    return this.meetupProposalsService.findOneForUser(id, req.user.sub);
  }

  @Put(':id/status')
  @ApiOperation({ summary: 'Accept or decline a pending proposal as its recipient' })
  updateStatus(
    @Param('id') id: string,
    @Request() req: AuthenticatedRequest,
    @Body() dto: UpdateMeetupProposalDto,
  ) {
    return this.meetupProposalsService.updateStatus(id, req.user.sub, dto);
  }

  @Put(':id/complete')
  @ApiOperation({ summary: 'Submit participant-specific post-meetup outcome feedback' })
  complete(
    @Param('id') id: string,
    @Request() req: AuthenticatedRequest,
    @Body() dto: CompleteMeetupDto,
  ) {
    return this.meetupProposalsService.complete(id, req.user.sub, dto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Cancel an active meetup proposal as a participant' })
  remove(@Param('id') id: string, @Request() req: AuthenticatedRequest) {
    return this.meetupProposalsService.remove(id, req.user.sub);
  }
}
