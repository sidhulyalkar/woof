import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  UseGuards,
  Request,
  Query,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth, ApiQuery } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { EventsService } from './events.service';
import { CreateEventDto } from './dto/create-event.dto';
import { UpdateEventDto } from './dto/update-event.dto';
import { CreateRSVPDto, EventFeedbackDto } from './dto/rsvp-event.dto';

@ApiTags('events')
@Controller('events')
export class EventsController {
  constructor(private readonly eventsService: EventsService) {}

  @Post()
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Create a new community event' })
  @ApiResponse({ status: 201, description: 'Event created successfully' })
  async create(@Request() req: AuthenticatedRequest, @Body() createEventDto: CreateEventDto) {
    return this.eventsService.create(req.user.sub, createEventDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all events with optional filters' })
  @ApiQuery({ name: 'type', required: false, description: 'Filter by event type' })
  @ApiQuery({ name: 'upcoming', required: false, description: 'Show only upcoming events' })
  @ApiResponse({ status: 200, description: 'List of events' })
  async findAll(@Query('type') type?: string, @Query('upcoming') upcoming?: string) {
    const upcomingBool = upcoming === 'true';
    return this.eventsService.findAll(type, upcomingBool);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a specific event with details' })
  @ApiResponse({ status: 200, description: 'Event details' })
  @ApiResponse({ status: 404, description: 'Event not found' })
  async findOne(@Param('id') id: string) {
    return this.eventsService.findOne(id);
  }

  @Patch(':id')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Update an event (organizer only)' })
  @ApiResponse({ status: 200, description: 'Event updated successfully' })
  @ApiResponse({ status: 400, description: 'Not authorized or event not found' })
  async update(
    @Param('id') id: string,
    @Request() req: AuthenticatedRequest,
    @Body() updateEventDto: UpdateEventDto,
  ) {
    return this.eventsService.update(id, req.user.sub, updateEventDto);
  }

  @Delete(':id')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Delete an event (organizer only)' })
  @ApiResponse({ status: 200, description: 'Event deleted successfully' })
  @ApiResponse({ status: 400, description: 'Not authorized or event not found' })
  async remove(@Param('id') id: string, @Request() req: AuthenticatedRequest) {
    return this.eventsService.remove(id, req.user.sub);
  }

  @Post(':id/rsvp')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'RSVP to an event' })
  @ApiResponse({ status: 201, description: 'RSVP created/updated successfully' })
  @ApiResponse({ status: 400, description: 'Event full or other error' })
  async rsvp(
    @Param('id') id: string,
    @Request() req: AuthenticatedRequest,
    @Body() createRSVPDto: CreateRSVPDto,
  ) {
    return this.eventsService.rsvp(id, req.user.sub, createRSVPDto);
  }

  @Get('rsvps/me')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get all RSVPs for the current user' })
  @ApiResponse({ status: 200, description: 'List of user RSVPs' })
  async getUserRSVPs(@Request() req: AuthenticatedRequest) {
    return this.eventsService.getUserRSVPs(req.user.sub);
  }

  @Post(':id/feedback')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Submit feedback for an event' })
  @ApiResponse({ status: 201, description: 'Feedback submitted successfully' })
  @ApiResponse({ status: 400, description: 'Must RSVP to submit feedback' })
  async submitFeedback(
    @Param('id') id: string,
    @Request() req: AuthenticatedRequest,
    @Body() eventFeedbackDto: EventFeedbackDto,
  ) {
    return this.eventsService.submitFeedback(id, req.user.sub, eventFeedbackDto);
  }

  @Get(':id/feedback')
  @ApiOperation({ summary: 'Get feedback for an event' })
  @ApiResponse({ status: 200, description: 'Event feedback with averages' })
  async getEventFeedback(@Param('id') id: string) {
    return this.eventsService.getEventFeedback(id);
  }

  @Post(':id/check-in')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Check in to an event (awards points)' })
  @ApiResponse({ status: 200, description: 'Checked in successfully, points awarded' })
  @ApiResponse({ status: 400, description: 'Must RSVP first or already checked in' })
  async checkIn(@Param('id') id: string, @Request() req: AuthenticatedRequest) {
    return this.eventsService.checkIn(id, req.user.sub);
  }
}
