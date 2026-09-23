import {
  BadRequestException,
  ConflictException,
  ForbiddenException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { Prisma, type MeetupOutcome } from '@woof/database';
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
        conversation.participants.some((participant) => participant.userId === dto.recipientId)
    );
    if (!directConversation) {
      throw new BadRequestException(
        'Start a two-person conversation before proposing an in-person meetup'
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
          ...(dto.suggestedVenue.area?.trim() ? { area: dto.suggestedVenue.area.trim() } : {}),
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
    const proposalIds = [...sent, ...received].map((proposal) => proposal.id);
    const outcomes =
      proposalIds.length === 0
        ? []
        : await this.prisma.meetupOutcome.findMany({
            where: { participantId: userId, proposalId: { in: proposalIds } },
            orderBy: { createdAt: 'desc' },
          });
    return { sent, received, outcomes };
  }

  async findOneForUser(id: string, userId: string) {
    const proposal = await this.prisma.meetupProposal.findUnique({ where: { id } });
    if (!proposal) throw new NotFoundException(`Meetup proposal ${id} not found`);
    if (proposal.proposerId !== userId && proposal.recipientId !== userId) {
      throw new NotFoundException(`Meetup proposal ${id} not found`);
    }
    return proposal;
  }

  async findOutcomeForUser(id: string, userId: string) {
    await this.findOneForUser(id, userId);
    return this.prisma.meetupOutcome.findUnique({
      where: { proposalId_participantId: { proposalId: id, participantId: userId } },
    });
  }

  async updateStatus(id: string, userId: string, dto: UpdateMeetupProposalDto) {
    const proposal = await this.findOneForUser(id, userId);
    if (proposal.recipientId !== userId) {
      throw new ForbiddenException('Only the recipient can accept or decline this proposal');
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

    const transition = await this.prisma.meetupProposal.updateMany({
      where: {
        id,
        recipientId: userId,
        status: MeetupProposalStatus.PENDING,
      },
      data: { status: dto.status },
    });
    if (transition.count !== 1) {
      throw new ConflictException('This meetup proposal is no longer pending');
    }

    const updated = await this.prisma.meetupProposal.findUnique({ where: { id } });
    if (!updated) throw new NotFoundException(`Meetup proposal ${id} not found`);

    await this.recordTelemetryBestEffort(
      userId,
      dto.status === MeetupProposalStatus.ACCEPTED ? 'MEETUP_ACCEPTED' : 'MEETUP_DECLINED',
      { proposalId: id, otherUserId: proposal.proposerId }
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

    const normalized = this.normalizeOutcome(id, userId, dto);
    let outcome: MeetupOutcome;
    let currentProposal = proposal;
    let idempotentRetry = false;
    let created = false;

    try {
      const result = await this.prisma.$transaction(async (tx) => {
        const createdOutcome = await tx.meetupOutcome.create({ data: normalized });

        if (createdOutcome.occurred && proposal.status === MeetupProposalStatus.ACCEPTED) {
          await tx.meetupProposal.updateMany({
            where: { id, status: MeetupProposalStatus.ACCEPTED },
            data: {
              status: MeetupProposalStatus.COMPLETED,
              occurredAt: proposal.occurredAt ?? new Date(),
            },
          });
        }

        const latestProposal = await tx.meetupProposal.findUnique({ where: { id } });
        if (!latestProposal) throw new NotFoundException(`Meetup proposal ${id} not found`);
        return { outcome: createdOutcome, proposal: latestProposal };
      });
      outcome = result.outcome;
      currentProposal = result.proposal;
      created = true;
    } catch (error) {
      if (!this.isUniqueViolation(error)) throw error;
      const existing = await this.prisma.meetupOutcome.findUnique({
        where: { proposalId_participantId: { proposalId: id, participantId: userId } },
      });
      if (!existing || !this.outcomeMatches(existing, normalized)) {
        throw new ConflictException('You already submitted different feedback for this meetup');
      }
      outcome = existing;
      idempotentRetry = true;
      const latestProposal = await this.prisma.meetupProposal.findUnique({ where: { id } });
      if (latestProposal) currentProposal = latestProposal;
    }

    if (created) {
      await this.recordTelemetryBestEffort(userId, OUTCOME_EVENT, {
        proposalId: id,
        occurred: outcome.occurred,
      });
      if (outcome.checklistOk === false) {
        await this.recordTelemetryBestEffort(userId, 'MEETUP_SAFETY_CONCERN_RECORDED', {
          proposalId: id,
        });
      }
    }

    return {
      proposal: currentProposal,
      outcome,
      feedbackRecorded: true as const,
      idempotentRetry,
      reportSuggested: outcome.checklistOk === false,
      repeatPlanningEligible:
        outcome.occurred && (outcome.meetAgain === 'yes' || outcome.meetAgain === 'maybe'),
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
    await this.recordTelemetryBestEffort(userId, 'MEETUP_CANCELLED', { proposalId: id });
    return updated;
  }

  remove(id: string, userId: string) {
    return this.cancel(id, userId);
  }

  async getStats(userId: string) {
    const [proposals, outcomes] = await Promise.all([
      this.prisma.meetupProposal.findMany({
        where: { OR: [{ proposerId: userId }, { recipientId: userId }] },
      }),
      this.prisma.meetupOutcome.findMany({
        where: { participantId: userId, rating: { not: null } },
        select: { rating: true },
      }),
    ]);
    return {
      total: proposals.length,
      pending: proposals.filter((proposal) => proposal.status === 'pending').length,
      accepted: proposals.filter((proposal) => proposal.status === 'accepted').length,
      completed: proposals.filter((proposal) => proposal.status === 'completed').length,
      avgRating:
        outcomes.length > 0
          ? outcomes.reduce((sum, outcome) => sum + (outcome.rating ?? 0), 0) / outcomes.length
          : 0,
    };
  }

  private normalizeOutcome(proposalId: string, participantId: string, dto: CompleteMeetupDto) {
    const structuredTags = [
      dto.dogExperience ? `dog_${dto.dogExperience}` : null,
      dto.ownerExperience ? `owner_${dto.ownerExperience}` : null,
      dto.meetAgain ? `meet_again_${dto.meetAgain}` : null,
    ].filter((tag): tag is string => tag !== null);
    const feedbackTags = [...structuredTags, ...(dto.feedbackTags ?? [])]
      .map((tag) =>
        tag
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9_-]+/g, '_')
      )
      .filter(Boolean);
    return {
      proposalId,
      participantId,
      occurred: dto.occurred,
      dogExperience: dto.dogExperience ?? null,
      ownerExperience: dto.ownerExperience ?? null,
      meetAgain: dto.meetAgain ?? null,
      rating: dto.rating ?? null,
      feedbackTags: [...new Set(feedbackTags)].sort().slice(0, 16),
      checklistOk: dto.checklistOk ?? null,
      notes: dto.notes?.trim() || null,
    };
  }

  private outcomeMatches(
    existing: {
      occurred: boolean;
      dogExperience: string | null;
      ownerExperience: string | null;
      meetAgain: string | null;
      rating: number | null;
      feedbackTags: string[];
      checklistOk: boolean | null;
      notes: string | null;
    },
    expected: ReturnType<MeetupProposalsService['normalizeOutcome']>
  ) {
    return (
      existing.occurred === expected.occurred &&
      existing.dogExperience === expected.dogExperience &&
      existing.ownerExperience === expected.ownerExperience &&
      existing.meetAgain === expected.meetAgain &&
      existing.rating === expected.rating &&
      existing.checklistOk === expected.checklistOk &&
      existing.notes === expected.notes &&
      existing.feedbackTags.length === expected.feedbackTags.length &&
      existing.feedbackTags.every((tag, index) => tag === expected.feedbackTags[index])
    );
  }

  private isUniqueViolation(error: unknown): boolean {
    return error instanceof Prisma.PrismaClientKnownRequestError && error.code === 'P2002';
  }

  private async recordTelemetryBestEffort(
    userId: string,
    event: string,
    data: Prisma.InputJsonObject
  ) {
    try {
      await this.recordTelemetry(userId, event, data);
    } catch {
      // Canonical meetup outcomes must not fail because observability is degraded.
    }
  }

  private async recordTelemetry(userId: string, event: string, data: Prisma.InputJsonObject) {
    await this.prisma.telemetry.create({
      data: { userId, source: 'meetup', event, data },
    });
  }
}
