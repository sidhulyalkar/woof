import {
  Controller,
  Get,
  Post,
  Put,
  Delete,
  Body,
  Param,
  UseGuards,
  Request,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { MeetupProposalsService } from './meetup-proposals.service';
import { CreateMeetupProposalDto } from './dto/create-meetup-proposal.dto';
import { UpdateMeetupProposalDto, CompleteMeetupDto } from './dto/update-meetup-proposal.dto';

@ApiTags('meetup-proposals')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('meetup-proposals')
export class MeetupProposalsController {
  constructor(private readonly meetupProposalsService: MeetupProposalsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a new meetup proposal' })
  @ApiResponse({ status: 201, description: 'Meetup proposal created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async create(@Request() req: any, @Body() createMeetupProposalDto: CreateMeetupProposalDto) {
    return this.meetupProposalsService.create(req.user.id, createMeetupProposalDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all meetup proposals for the current user' })
  @ApiResponse({ status: 200, description: 'List of sent and received proposals' })
  async findAll(@Request() req: any) {
    return this.meetupProposalsService.findAllForUser(req.user.id);
  }

  @Get('stats')
  @ApiOperation({ summary: 'Get meetup statistics for the current user' })
  @ApiResponse({ status: 200, description: 'Meetup statistics' })
  async getStats(@Request() req: any) {
    return this.meetupProposalsService.getStats(req.user.id);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a specific meetup proposal' })
  @ApiResponse({ status: 200, description: 'Meetup proposal details' })
  @ApiResponse({ status: 404, description: 'Proposal not found' })
  async findOne(@Param('id') id: string) {
    return this.meetupProposalsService.findOne(id);
  }

  @Put(':id/status')
  @ApiOperation({ summary: 'Update proposal status (accept/decline)' })
  @ApiResponse({ status: 200, description: 'Status updated successfully' })
  @ApiResponse({ status: 400, description: 'Invalid status or not authorized' })
  async updateStatus(
    @Param('id') id: string,
    @Request() req: any,
    @Body() updateMeetupProposalDto: UpdateMeetupProposalDto,
  ) {
    return this.meetupProposalsService.updateStatus(id, req.user.id, updateMeetupProposalDto);
  }

  @Put(':id/complete')
  @ApiOperation({ summary: 'Mark meetup as completed with feedback' })
  @ApiResponse({ status: 200, description: 'Meetup completed successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input or not authorized' })
  async complete(@Param('id') id: string, @Request() req: any, @Body() completeMeetupDto: CompleteMeetupDto) {
    return this.meetupProposalsService.complete(id, req.user.id, completeMeetupDto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Cancel a meetup proposal' })
  @ApiResponse({ status: 200, description: 'Proposal cancelled successfully' })
  @ApiResponse({ status: 400, description: 'Not authorized' })
  async remove(@Param('id') id: string, @Request() req: any) {
    return this.meetupProposalsService.remove(id, req.user.id);
  }
}
