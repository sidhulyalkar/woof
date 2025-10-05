import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateMeetupProposalDto } from './dto/create-meetup-proposal.dto';
import { UpdateMeetupProposalDto, CompleteMeetupDto } from './dto/update-meetup-proposal.dto';

@Injectable()
export class MeetupProposalsService {
  constructor(private prisma: PrismaService) {}

  /**
   * Create a new meetup proposal
   */
  async create(proposerId: string, dto: CreateMeetupProposalDto) {
    return this.prisma.meetupProposal.create({
      data: {
        proposerId,
        recipientId: dto.recipientId,
        suggestedTime: new Date(dto.suggestedTime),
        suggestedVenue: dto.suggestedVenue,
        notes: dto.notes,
      },
    });
  }

  /**
   * Get all proposals for a user (sent + received)
   */
  async findAllForUser(userId: string) {
    const [sent, received] = await Promise.all([
      this.prisma.meetupProposal.findMany({
        where: { proposerId: userId },
        orderBy: { createdAt: 'desc' },
      }),
      this.prisma.meetupProposal.findMany({
        where: { recipientId: userId },
        orderBy: { createdAt: 'desc' },
      }),
    ]);

    return { sent, received };
  }

  /**
   * Get a specific proposal by ID
   */
  async findOne(id: string) {
    const proposal = await this.prisma.meetupProposal.findUnique({
      where: { id },
    });

    if (!proposal) {
      throw new NotFoundException(`Meetup proposal ${id} not found`);
    }

    return proposal;
  }

  /**
   * Accept or decline a proposal
   */
  async updateStatus(id: string, userId: string, dto: UpdateMeetupProposalDto) {
    const proposal = await this.findOne(id);

    // Only recipient can accept/decline
    if (proposal.recipientId !== userId) {
      throw new BadRequestException('Only the recipient can update this proposal');
    }

    return this.prisma.meetupProposal.update({
      where: { id },
      data: { status: dto.status },
    });
  }

  /**
   * Complete a meetup with feedback
   */
  async complete(id: string, userId: string, dto: CompleteMeetupDto) {
    const proposal = await this.findOne(id);

    // Both parties can mark complete
    if (proposal.proposerId !== userId && proposal.recipientId !== userId) {
      throw new BadRequestException('You are not part of this meetup');
    }

    return this.prisma.meetupProposal.update({
      where: { id },
      data: {
        status: 'completed',
        occurredAt: dto.occurred ? new Date() : null,
        rating: dto.rating,
        feedbackTags: dto.feedbackTags || [],
        checklistOk: dto.checklistOk,
        notes: dto.notes,
      },
    });
  }

  /**
   * Cancel a proposal
   */
  async cancel(id: string, userId: string) {
    const proposal = await this.findOne(id);

    // Both parties can cancel
    if (proposal.proposerId !== userId && proposal.recipientId !== userId) {
      throw new BadRequestException('You are not part of this meetup');
    }

    return this.prisma.meetupProposal.update({
      where: { id },
      data: { status: 'cancelled' },
    });
  }

  /**
   * Delete a proposal (soft delete by setting status to cancelled)
   */
  async remove(id: string, userId: string) {
    return this.cancel(id, userId);
  }

  /**
   * Get meetup statistics for a user
   */
  async getStats(userId: string) {
    const proposals = await this.prisma.meetupProposal.findMany({
      where: {
        OR: [{ proposerId: userId }, { recipientId: userId }],
      },
    });

    const stats = {
      total: proposals.length,
      pending: proposals.filter((p: any) => p.status === 'pending').length,
      accepted: proposals.filter((p: any) => p.status === 'accepted').length,
      completed: proposals.filter((p: any) => p.status === 'completed').length,
      avgRating:
        proposals
          .filter((p: any) => p.rating !== null)
          .reduce((sum: number, p: any) => sum + (p.rating || 0), 0) /
          proposals.filter((p: any) => p.rating !== null).length || 0,
    };

    return stats;
  }
}
