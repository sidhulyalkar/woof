import {
  BadRequestException,
  ForbiddenException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateMeetupProposalDto } from './dto/create-meetup-proposal.dto';
import {
  CompleteMeetupDto,
  MeetupProposalStatus,
  UpdateMeetupProposalDto,
} from './dto/update-meetup-proposal.dto';

const OUTCOME_EVENT = 'MEETUP_OUTCOME_REPORTED';

@Injectable()
export class MeetupProposalsService {
  constructor(private readonly prisma: PrismaService) {}

  async create(proposerId: string, dto: CreateMeetupProposalDto) {
    if (proposerId === dto.recipientId) {
      throw new BadRequestException('You cannot propose a meetup to yourself');
    }

    const suggestedTime = new Date(dto.suggestedTime);
    const minTime = Date.now() + 15 * 60 * 1000;
    const maxTime = Date.now() + 30 * 24 * 60 * 60 * 1000;
    if (suggestedTime.getTime() < minTime || suggestedTime.getTime() > maxTime) {
      throw new BadRequestException('Meetups must be proposed 15 minutes to 30 days in advance');
    }

    const [recipient, blocked, conversations] = await Promise.all([
      this.prisma.user.findUnique({
        where: { id: dto.recipientId },
        select: { id: true },
      }),
      this.prisma.blockedUser.findFirst({
        where: {
          OR: [
            { userId: proposerId, blockedId: dto.recipientId },
            { userId: dto.recipientId, blockedId: proposerId },
          ],
        },
        select: { id: true },
      }),
      this.prisma.conversation.findMany({
        where: {
          AND: [
            { participants: { some: { userId: proposerId } } },
            { participants: { some: { userId: dto.recipientId } } },
            { messages: { some: {} } },
          ],
        },
        select: {
          id: true,
          participants: { select: { userId: true } },
        },
        take: 20,
      }),
    ]);

    if (!recipient) throw new NotFoundException('Member not found');
    if (blocked) throw new ForbiddenException('Meetup coordination is unavailable for this pair');
    const directConversation = conversations.find(
      (conversation) =>
        conversation.participants.length === 2 &&
        conversation.participants.some((participant) => participant.userId === proposerId) &&
        conversation.participants.some((participant) => participant.userId === dto.recipientId),
    );
    if (!directConversation) {
      throw new BadRequestException(
        'Start a two-person conversation before proposing an in-person meetup',
      );
    }

    const proposal = await this.prisma.meetupProposal.create({
      data: {
        proposerId,
        recipientId: dto.recipientId,
        suggestedTime,
        suggestedVenue: {
          name: dto.suggestedVenue.name.trim(),
          type: dto.suggestedVenue.type.trim(),
          ...(dto.suggestedVenue.area?.trim()
            ? { area: dto.suggestedVenue.area.trim() }
            : {}),
        },
        notes: dto.notes?.trim() || null,
      },
    });

    await this.recordTelemetry(proposerId, 'MEETUP_PROPOSED', {
      proposalId: proposal.id,
      recipientId: dto.recipientId,
      conversationId: directConversation.id,
    });
    return proposal;
  }

  async findAllForUser(userId: string) {
    const [sent, received] = await Promise.all([
      this.prisma.meetupProposal.findMany({
        where: { proposerId: userId },
        orderBy: { createdAt: 'desc' },
        take: 100,
      }),
      this.prisma.meetupProposal.findMany({
        where: { recipientId: userId },
        orderBy: { createdAt: 'desc' },
        take: 100,
      }),
    ]);
    return { sent, received };
  }

  async findOneForUser(id: string, userId: string) {
    const proposal = await this.prisma.meetupProposal.findUnique({ where: { id } });
    if (!proposal) throw new NotFoundException(`Meetup proposal ${id} not found`);
    if (proposal.proposerId !== userId && proposal.recipientId !== userId) {
      throw new NotFoundException(`Meetup proposal ${id} not found`);
    }
    return proposal;
  }

  async updateStatus(id: string, userId: string, dto: UpdateMeetupProposalDto) {
    const proposal = await this.findOneForUser(id, userId);
    if (proposal.recipientId !== userId) {
      throw new ForbiddenException('Only the recipient can accept or decline this proposal');
    }
    if (proposal.status !== MeetupProposalStatus.PENDING) {
      throw new BadRequestException('Only pending proposals can be accepted or declined');
    }
    if (![MeetupProposalStatus.ACCEPTED, MeetupProposalStatus.DECLINED].includes(dto.status)) {
      throw new BadRequestException('Status must be accepted or declined');
    }

    const blocked = await this.prisma.blockedUser.findFirst({
      where: {
        OR: [
          { userId: proposal.proposerId, blockedId: proposal.recipientId },
          { userId: proposal.recipientId, blockedId: proposal.proposerId },
        ],
      },
      select: { id: true },
    });
    if (blocked) throw new ForbiddenException('Meetup coordination is unavailable for this pair');

    const updated = await this.prisma.meetupProposal.update({
      where: { id },
      data: { status: dto.status },
    });
    await this.recordTelemetry(
      userId,
      dto.status === MeetupProposalStatus.ACCEPTED ? 'MEETUP_ACCEPTED' : 'MEETUP_DECLINED',
      { proposalId: id, otherUserId: proposal.proposerId },
    );
    return updated;
  }

  async complete(id: string, userId: string, dto: CompleteMeetupDto) {
    const proposal = await this.findOneForUser(id, userId);
    if (
      proposal.status !== MeetupProposalStatus.ACCEPTED &&
      proposal.status !== MeetupProposalStatus.COMPLETED
    ) {
      throw new BadRequestException('Only accepted meetups can receive outcome feedback');
    }

    const existingFeedback = await this.prisma.telemetry.findFirst({
      where: {
        userId,
        event: OUTCOME_EVENT,
        data: { path: ['proposalId'], equals: id },
      },
      select: { id: true },
    });
    if (existingFeedback) {
      throw new BadRequestException('You already submitted feedback for this meetup');
    }

    const safeTags = (dto.feedbackTags ?? [])
      .map((tag) => tag.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_'))
      .filter(Boolean);
    await this.recordTelemetry(userId, OUTCOME_EVENT, {
      proposalId: id,
      otherUserId:
        proposal.proposerId === userId ? proposal.recipientId : proposal.proposerId,
      occurred: dto.occurred,
      rating: dto.rating ?? null,
      feedbackTags: safeTags,
      checklistOk: dto.checklistOk ?? null,
    });

    const previousRating = proposal.rating;
    const aggregateRating =
      dto.rating === undefined
        ? previousRating
        : previousRating === null
          ? dto.rating
          : Math.round(((previousRating + dto.rating) / 2) * 10) / 10;
    const mergedTags = [...new Set([...(proposal.feedbackTags ?? []), ...safeTags])].slice(0, 16);

    const updated = await this.prisma.meetupProposal.update({
      where: { id },
      data: {
        status: dto.occurred
          ? MeetupProposalStatus.COMPLETED
          : MeetupProposalStatus.CANCELLED,
        occurredAt: dto.occurred ? (proposal.occurredAt ?? new Date()) : null,
        rating: aggregateRating,
        feedbackTags: mergedTags,
        checklistOk:
          dto.checklistOk === undefined
            ? proposal.checklistOk
            : proposal.checklistOk === false
              ? false
              : dto.checklistOk,
        notes: proposal.notes,
      },
    });

    if (dto.checklistOk === false) {
      await this.recordTelemetry(userId, 'MEETUP_SAFETY_CONCERN_RECORDED', {
        proposalId: id,
      });
    }

    return {
      proposal: updated,
      feedbackRecorded: true,
      reportSuggested: dto.checklistOk === false,
    };
  }

  async cancel(id: string, userId: string) {
    const proposal = await this.findOneForUser(id, userId);
    if (
      proposal.status === MeetupProposalStatus.COMPLETED ||
      proposal.status === MeetupProposalStatus.DECLINED
    ) {
      throw new BadRequestException('This meetup can no longer be cancelled');
    }
    const updated = await this.prisma.meetupProposal.update({
      where: { id },
      data: { status: MeetupProposalStatus.CANCELLED },
    });
    await this.recordTelemetry(userId, 'MEETUP_CANCELLED', { proposalId: id });
    return updated;
  }

  remove(id: string, userId: string) {
    return this.cancel(id, userId);
  }

  async getStats(userId: string) {
    const proposals = await this.prisma.meetupProposal.findMany({
      where: { OR: [{ proposerId: userId }, { recipientId: userId }] },
    });
    const rated = proposals.filter((proposal) => proposal.rating !== null);
    return {
      total: proposals.length,
      pending: proposals.filter((proposal) => proposal.status === 'pending').length,
      accepted: proposals.filter((proposal) => proposal.status === 'accepted').length,
      completed: proposals.filter((proposal) => proposal.status === 'completed').length,
      avgRating:
        rated.length > 0
          ? rated.reduce((sum, proposal) => sum + (proposal.rating ?? 0), 0) / rated.length
          : 0,
    };
  }

  private async recordTelemetry(
    userId: string,
    event: string,
    data: Record<string, unknown>,
  ) {
    await this.prisma.telemetry.create({
      data: { userId, source: 'meetup', event, data: data as any },
    });
  }
}
