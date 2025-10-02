import { Injectable, NotFoundException, ConflictException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { Prisma } from '@woof/database';

@Injectable()
export class MeetupsService {
  constructor(private prisma: PrismaService) {}

  // ============================================
  // MEETUPS
  // ============================================

  async create(data: Prisma.MeetupCreateInput) {
    return this.prisma.meetup.create({
      data,
      include: {
        creator: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
        _count: {
          select: {
            invites: true,
          },
        },
      },
    });
  }

  async findAll(skip = 0, take = 20, creatorUserId?: string, status?: string) {
    const where: Prisma.MeetupWhereInput = {};
    if (creatorUserId) where.creatorUserId = creatorUserId;
    if (status) where.status = status;

    const [meetups, total] = await Promise.all([
      this.prisma.meetup.findMany({
        where,
        skip,
        take,
        include: {
          creator: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
            },
          },
          _count: {
            select: {
              invites: true,
            },
          },
        },
        orderBy: {
          startsAt: 'desc',
        },
      }),
      this.prisma.meetup.count({ where }),
    ]);

    return { meetups, total, skip, take };
  }

  async findById(id: string) {
    const meetup = await this.prisma.meetup.findUnique({
      where: { id },
      include: {
        creator: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
            email: true,
          },
        },
        invites: {
          include: {
            user: {
              select: {
                id: true,
                handle: true,
                avatarUrl: true,
              },
            },
            pet: {
              select: {
                id: true,
                name: true,
                avatarUrl: true,
              },
            },
          },
        },
      },
    });

    if (!meetup) {
      throw new NotFoundException(`Meetup with ID ${id} not found`);
    }

    return meetup;
  }

  async update(id: string, data: Prisma.MeetupUpdateInput) {
    try {
      return await this.prisma.meetup.update({
        where: { id },
        data,
        include: {
          creator: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
            },
          },
          _count: {
            select: {
              invites: true,
            },
          },
        },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Meetup with ID ${id} not found`);
      }
      throw error;
    }
  }

  async delete(id: string) {
    try {
      return await this.prisma.meetup.delete({
        where: { id },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Meetup with ID ${id} not found`);
      }
      throw error;
    }
  }

  // ============================================
  // MEETUP INVITES
  // ============================================

  async createInvite(data: Prisma.MeetupInviteCreateInput) {
    return this.prisma.meetupInvite.create({
      data,
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
        meetup: {
          select: {
            id: true,
            title: true,
            startsAt: true,
          },
        },
      },
    });
  }

  async updateInvite(id: string, data: Prisma.MeetupInviteUpdateInput) {
    try {
      return await this.prisma.meetupInvite.update({
        where: { id },
        data,
        include: {
          user: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
            },
          },
          pet: {
            select: {
              id: true,
              name: true,
              avatarUrl: true,
            },
          },
        },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Meetup invite with ID ${id} not found`);
      }
      throw error;
    }
  }

  async deleteInvite(id: string) {
    try {
      return await this.prisma.meetupInvite.delete({
        where: { id },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Meetup invite with ID ${id} not found`);
      }
      throw error;
    }
  }

  async getMeetupInvites(meetupId: string) {
    return this.prisma.meetupInvite.findMany({
      where: { meetupId },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
  }

  async getUserMeetupInvites(userId: string) {
    return this.prisma.meetupInvite.findMany({
      where: { userId },
      include: {
        meetup: {
          include: {
            creator: {
              select: {
                id: true,
                handle: true,
                avatarUrl: true,
              },
            },
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
  }
}
