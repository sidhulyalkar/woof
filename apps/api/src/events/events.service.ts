import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { CreateEventDto } from './dto/create-event.dto';
import { UpdateEventDto } from './dto/update-event.dto';
import { CreateRSVPDto, EventFeedbackDto } from './dto/rsvp-event.dto';

// Community events acknowledge participation without issuing legacy totalPoints.
@Injectable()
export class EventsService {
  constructor(private prisma: PrismaService) {}

  async create(hostUserId: string, dto: CreateEventDto) {
    return this.prisma.communityEvent.create({
      data: {
        hostUserId,
        title: dto.title,
        description: dto.description,
        venueType: dto.type || 'park',
        startTime: new Date(dto.startTime),
        endTime: dto.endTime
          ? new Date(dto.endTime)
          : new Date(new Date(dto.startTime).getTime() + 2 * 60 * 60 * 1000),
        venueName: dto.locationName,
        address: dto.locationName,
        lat: dto.lat,
        lng: dto.lng,
        capacity: dto.maxAttendees,
      },
    });
  }

  async findAll(type?: string, upcoming?: boolean) {
    const where: Prisma.CommunityEventWhereInput = {};

    if (type) {
      where.venueType = type;
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

  async update(id: string, userId: string, dto: UpdateEventDto) {
    const event = await this.findOne(id);

    if (event.hostUserId !== userId) {
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

  async remove(id: string, userId: string) {
    const event = await this.findOne(id);

    if (event.hostUserId !== userId) {
      throw new BadRequestException('Only the organizer can delete this event');
    }

    return this.prisma.communityEvent.delete({
      where: { id },
    });
  }

  async rsvp(eventId: string, userId: string, dto: CreateRSVPDto) {
    const event = await this.findOne(eventId);

    if (event.capacity && dto.status === 'going') {
      const goingCount = event.rsvps.filter((rsvp) => rsvp.status === 'going').length;
      if (goingCount >= event.capacity) {
        throw new BadRequestException('Event is full');
      }
    }

    const existingRSVP = await this.prisma.eventRSVP.findUnique({
      where: {
        eventId_userId: {
          eventId,
          userId,
        },
      },
    });

    if (existingRSVP) {
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

    return this.prisma.eventRSVP.create({
      data: {
        eventId,
        userId,
        status: dto.status,
      },
    });
  }

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

  async checkIn(eventId: string, userId: string) {
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

    if (rsvp.checkedInAt) {
      throw new BadRequestException('You have already checked in to this event');
    }

    const updatedRSVP = await this.prisma.eventRSVP.update({
      where: {
        eventId_userId: {
          eventId,
          userId,
        },
      },
      data: {
        checkedInAt: new Date(),
      },
    });

    return {
      ...updatedRSVP,
      message: 'Checked in successfully. Thanks for joining the community event.',
    };
  }

  async submitFeedback(eventId: string, userId: string, dto: EventFeedbackDto) {
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
          surfaceType: dto.surfaceType,
          crowding: dto.crowding,
          noiseLevel: dto.noiseLevel,
          tags: dto.tags || [],
          notes: dto.notes,
        },
      });
    } else {
      feedback = await this.prisma.eventFeedback.create({
        data: {
          eventId,
          userId,
          vibeScore: dto.vibeScore,
          petDensity: dto.petDensity,
          surfaceType: dto.surfaceType,
          crowding: dto.crowding,
          noiseLevel: dto.noiseLevel,
          tags: dto.tags || [],
          notes: dto.notes,
        },
      });
      isNewFeedback = true;
    }

    return {
      ...feedback,
      message: isNewFeedback
        ? 'Feedback submitted. Thanks for helping the community learn about this event.'
        : 'Feedback updated successfully.',
    };
  }

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

    const avgVibeScore =
      feedback.reduce((sum, item) => sum + item.vibeScore, 0) / feedback.length || 0;

    return {
      feedback,
      averages: {
        vibeScore: avgVibeScore,
      },
      totalFeedback: feedback.length,
    };
  }
}
