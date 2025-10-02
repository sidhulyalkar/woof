import { Controller, Get, Post, Put, Delete, Param, Body, Query, UseGuards } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { MeetupsService } from './meetups.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { Prisma } from '@woof/database';

@ApiTags('meetups')
@Controller('meetups')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class MeetupsController {
  constructor(private meetupsService: MeetupsService) {}

  // ============================================
  // MEETUPS
  // ============================================

  @Post()
  @ApiOperation({ summary: 'Create a new meetup' })
  @ApiResponse({ status: 201, description: 'Meetup created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async create(@Body() createMeetupDto: Prisma.MeetupCreateInput) {
    return this.meetupsService.create(createMeetupDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all meetups (paginated)' })
  @ApiResponse({ status: 200, description: 'List of meetups' })
  async findAll(
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('creatorUserId') creatorUserId?: string,
    @Query('status') status?: string,
  ) {
    return this.meetupsService.findAll(skip, take, creatorUserId, status);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get meetup by ID' })
  @ApiResponse({ status: 200, description: 'Meetup found' })
  @ApiResponse({ status: 404, description: 'Meetup not found' })
  async findOne(@Param('id') id: string) {
    return this.meetupsService.findById(id);
  }

  @Put(':id')
  @ApiOperation({ summary: 'Update meetup by ID' })
  @ApiResponse({ status: 200, description: 'Meetup updated successfully' })
  @ApiResponse({ status: 404, description: 'Meetup not found' })
  async update(
    @Param('id') id: string,
    @Body() updateMeetupDto: Prisma.MeetupUpdateInput,
  ) {
    return this.meetupsService.update(id, updateMeetupDto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete meetup by ID' })
  @ApiResponse({ status: 200, description: 'Meetup deleted successfully' })
  @ApiResponse({ status: 404, description: 'Meetup not found' })
  async delete(@Param('id') id: string) {
    return this.meetupsService.delete(id);
  }

  // ============================================
  // MEETUP INVITES
  // ============================================

  @Post('invites')
  @ApiOperation({ summary: 'Create a meetup invite' })
  @ApiResponse({ status: 201, description: 'Invite created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async createInvite(@Body() createInviteDto: Prisma.MeetupInviteCreateInput) {
    return this.meetupsService.createInvite(createInviteDto);
  }

  @Get(':meetupId/invites')
  @ApiOperation({ summary: 'Get all invites for a meetup' })
  @ApiResponse({ status: 200, description: 'List of invites' })
  async getMeetupInvites(@Param('meetupId') meetupId: string) {
    return this.meetupsService.getMeetupInvites(meetupId);
  }

  @Get('invites/user/:userId')
  @ApiOperation({ summary: 'Get all meetup invites for a user' })
  @ApiResponse({ status: 200, description: 'List of user invites' })
  async getUserMeetupInvites(@Param('userId') userId: string) {
    return this.meetupsService.getUserMeetupInvites(userId);
  }

  @Put('invites/:id')
  @ApiOperation({ summary: 'Update meetup invite (RSVP, check-in)' })
  @ApiResponse({ status: 200, description: 'Invite updated successfully' })
  @ApiResponse({ status: 404, description: 'Invite not found' })
  async updateInvite(
    @Param('id') id: string,
    @Body() updateInviteDto: Prisma.MeetupInviteUpdateInput,
  ) {
    return this.meetupsService.updateInvite(id, updateInviteDto);
  }

  @Delete('invites/:id')
  @ApiOperation({ summary: 'Delete meetup invite' })
  @ApiResponse({ status: 200, description: 'Invite deleted successfully' })
  @ApiResponse({ status: 404, description: 'Invite not found' })
  async deleteInvite(@Param('id') id: string) {
    return this.meetupsService.deleteInvite(id);
  }
}
