import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { GamificationService } from '../gamification/gamification.service';
import { CreateEventDto } from './dto/create-event.dto';
import { UpdateEventDto } from './dto/update-event.dto';
import { CreateRSVPDto, UpdateRSVPDto, EventFeedbackDto } from './dto/rsvp-event.dto';

@Injectable()
export class EventsService {
  constructor(
    private prisma: PrismaService,
    private gamificationService: GamificationService,
  ) {}

  /**
   * Create a new community event
   */
  async create(organizerId: string, dto: CreateEventDto) {
    return this.prisma.communityEvent.create({
      data: {
        organizerId,
        title: dto.title,
        description: dto.description,
        type: dto.type,
        startTime: new Date(dto.startTime),
        endTime: dto.endTime ? new Date(dto.endTime) : null,
        locationName: dto.locationName,
        lat: dto.lat,
        lng: dto.lng,
        maxAttendees: dto.maxAttendees,
        tags: dto.tags || [],
      },
    });
  }

  /**
   * Get all events with optional filters
   */
  async findAll(type?: string, upcoming?: boolean) {
    const where: any = {};

    if (type) {
      where.type = type;
    }

    if (upcoming) {
      where.startTime = {
        gte: new Date(),
      };
    }

    return this.prisma.communityEvent.findMany({
      where,
      include: {
        organizer: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
        _count: {
          select: {
            rsvps: true,
          },
        },
      },
      orderBy: { startTime: 'asc' },
    });
  }

  /**
   * Get a specific event
   */
  async findOne(id: string) {
    const event = await this.prisma.communityEvent.findUnique({
      where: { id },
      include: {
        organizer: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
        rsvps: {
          include: {
            user: {
              select: {
                id: true,
                handle: true,
                avatarUrl: true,
              },
            },
          },
        },
      },
    });

    if (!event) {
      throw new NotFoundException(`Event ${id} not found`);
    }

    return event;
  }

  /**
   * Update an event
   */
  async update(id: string, userId: string, dto: UpdateEventDto) {
    const event = await this.findOne(id);

    // Only organizer can update
    if (event.organizerId !== userId) {
      throw new BadRequestException('Only the organizer can update this event');
    }

    return this.prisma.communityEvent.update({
      where: { id },
      data: {
        ...dto,
        startTime: dto.startTime ? new Date(dto.startTime) : undefined,
        endTime: dto.endTime ? new Date(dto.endTime) : undefined,
      },
    });
  }

  /**
   * Delete an event
   */
  async remove(id: string, userId: string) {
    const event = await this.findOne(id);

    // Only organizer can delete
    if (event.organizerId !== userId) {
      throw new BadRequestException('Only the organizer can delete this event');
    }

    return this.prisma.communityEvent.delete({
      where: { id },
    });
  }

  /**
   * Create or update RSVP
   */
  async rsvp(eventId: string, userId: string, dto: CreateRSVPDto) {
    const event = await this.findOne(eventId);

    // Check if max attendees reached
    if (event.maxAttendees && dto.status === 'going') {
      const goingCount = event.rsvps.filter((r: any) => r.status === 'going').length;
      if (goingCount >= event.maxAttendees) {
        throw new BadRequestException('Event is full');
      }
    }

    // Check if RSVP already exists
    const existingRSVP = await this.prisma.eventRSVP.findUnique({
      where: {
        eventId_userId: {
          eventId,
          userId,
        },
      },
    });

    if (existingRSVP) {
      // Update existing RSVP
      return this.prisma.eventRSVP.update({
        where: {
          eventId_userId: {
            eventId,
            userId,
          },
        },
        data: {
          status: dto.status,
        },
      });
    }

    // Create new RSVP
    return this.prisma.eventRSVP.create({
      data: {
        eventId,
        userId,
        status: dto.status,
      },
    });
  }

  /**
   * Get user's RSVPs
   */
  async getUserRSVPs(userId: string) {
    return this.prisma.eventRSVP.findMany({
      where: { userId },
      include: {
        event: {
          include: {
            organizer: {
              select: {
                id: true,
                handle: true,
                avatarUrl: true,
              },
            },
          },
        },
      },
      orderBy: {
        event: {
          startTime: 'asc',
        },
      },
    });
  }

  /**
   * Check in to an event (awards points)
   */
  async checkIn(eventId: string, userId: string) {
    // Check if user RSVP'd to the event
    const rsvp = await this.prisma.eventRSVP.findUnique({
      where: {
        eventId_userId: {
          eventId,
          userId,
        },
      },
    });

    if (!rsvp) {
      throw new BadRequestException('You must RSVP to this event before checking in');
    }

    if (rsvp.checkedIn) {
      throw new BadRequestException('You have already checked in to this event');
    }

    // Update RSVP to mark as checked in
    const updatedRSVP = await this.prisma.eventRSVP.update({
      where: {
        eventId_userId: {
          eventId,
          userId,
        },
      },
      data: {
        checkedIn: true,
        checkedInAt: new Date(),
      },
    });

    // Award points for attending event
    await this.gamificationService.awardPoints({
      userId,
      points: 5,
      reason: 'event_attended',
      relatedEntityId: eventId,
    });

    return {
      ...updatedRSVP,
      pointsAwarded: 5,
      message: 'Checked in successfully! You earned 5 points.',
    };
  }

  /**
   * Submit event feedback (awards points)
   */
  async submitFeedback(eventId: string, userId: string, dto: EventFeedbackDto) {
    // Check if user RSVP'd to the event
    const rsvp = await this.prisma.eventRSVP.findUnique({
      where: {
        eventId_userId: {
          eventId,
          userId,
        },
      },
    });

    if (!rsvp) {
      throw new BadRequestException('You must RSVP to this event to leave feedback');
    }

    // Check if feedback already exists
    const existingFeedback = await this.prisma.eventFeedback.findUnique({
      where: {
        eventId_userId: {
          eventId,
          userId,
        },
      },
    });

    let feedback;
    let isNewFeedback = false;

    if (existingFeedback) {
      // Update existing feedback
      feedback = await this.prisma.eventFeedback.update({
        where: {
          eventId_userId: {
            eventId,
            userId,
          },
        },
        data: {
          vibeScore: dto.vibeScore,
          petDensity: dto.petDensity,
          venueQuality: dto.venueQuality,
          tags: dto.tags || [],
          notes: dto.notes,
        },
      });
    } else {
      // Create new feedback
      feedback = await this.prisma.eventFeedback.create({
        data: {
          eventId,
          userId,
          vibeScore: dto.vibeScore,
          petDensity: dto.petDensity,
          venueQuality: dto.venueQuality,
          tags: dto.tags || [],
          notes: dto.notes,
        },
      });
      isNewFeedback = true;
    }

    // Award points for submitting feedback (only for new feedback)
    if (isNewFeedback) {
      await this.gamificationService.awardPoints({
        userId,
        points: 3,
        reason: 'event_feedback',
        relatedEntityId: eventId,
      });
    }

    return {
      ...feedback,
      pointsAwarded: isNewFeedback ? 3 : 0,
      message: isNewFeedback
        ? 'Feedback submitted! You earned 3 points.'
        : 'Feedback updated successfully.',
    };
  }

  /**
   * Get event feedback
   */
  async getEventFeedback(eventId: string) {
    const feedback = await this.prisma.eventFeedback.findMany({
      where: { eventId },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
      },
    });

    // Calculate averages
    const avgVibeScore =
      feedback.reduce((sum: number, f: any) => sum + f.vibeScore, 0) / feedback.length || 0;
    const avgPetDensity =
      feedback.reduce((sum: number, f: any) => sum + f.petDensity, 0) / feedback.length || 0;
    const avgVenueQuality =
      feedback.reduce((sum: number, f: any) => sum + f.venueQuality, 0) / feedback.length || 0;

    return {
      feedback,
      averages: {
        vibeScore: avgVibeScore,
        petDensity: avgPetDensity,
        venueQuality: avgVenueQuality,
      },
      totalFeedback: feedback.length,
    };
  }
}
