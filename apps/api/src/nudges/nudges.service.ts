import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { Prisma } from '@woof/database';
import { NotificationsService } from '../notifications/notifications.service';
import { PrismaService } from '../prisma/prisma.service';
import { CreateNudgeDto, NudgeReason, NudgeType } from './dto/create-nudge.dto';

const PAIR_COOLDOWN_HOURS = 48;
const DAILY_NUDGE_CAP = 2;
const DISMISSAL_LOOKBACK_DAYS = 14;

@Injectable()
export class NudgesService {
  private readonly logger = new Logger(NudgesService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly notificationsService: NotificationsService,
  ) {}

  /**
   * Precise proximity is intentionally not used for automatic beta nudges.
   * The schema can support future opt-in location experiences, but visibility
   * settings are not equivalent to explicit background/proximity consent.
   */
  @Cron(CronExpression.EVERY_5_MINUTES)
  async checkProximityNudges() {
    this.logger.debug(
      'Proximity nudges are disabled in beta until explicit location-discovery consent is modeled.',
    );
    return { enabled: false, reason: 'explicit-consent-required' };
  }

  /**
   * A conversation can create a quiet, in-app meetup suggestion for the actor.
   * Message volume alone is not treated as permission to push-notify both people.
   */
  async checkChatActivityNudges(conversationId: string, actorId: string) {
    const conversation = await this.prisma.conversation.findFirst({
      where: {
        id: conversationId,
        participants: { some: { userId: actorId } },
      },
      select: {
        id: true,
        participants: {
          select: {
            user: { select: { id: true, handle: true } },
          },
        },
        messages: {
          orderBy: { createdAt: 'desc' },
          take: 12,
          select: { id: true, senderId: true, createdAt: true },
        },
      },
    });

    if (!conversation || conversation.participants.length !== 2) {
      return { created: false, reason: 'conversation-not-eligible' };
    }

    const other = conversation.participants
      .map((participant) => participant.user)
      .find((user) => user.id !== actorId);
    if (!other) return { created: false, reason: 'conversation-not-eligible' };

    const distinctSenders = new Set(
      conversation.messages.map((message) => message.senderId),
    ).size;
    if (conversation.messages.length < 8 || distinctSenders < 2) {
      return { created: false, reason: 'insufficient-two-way-context' };
    }

    const canSend = await this.canSendNudge(actorId, other.id, NudgeType.MEETUP);
    if (!canSend) return { created: false, reason: 'cooldown-or-fatigue-cap' };

    const nudge = await this.createNudge(
      {
        userId: actorId,
        type: NudgeType.MEETUP,
        context: {
          targetUserId: other.id,
          reason: NudgeReason.CHAT_ACTIVITY,
          message: `Want to turn the conversation with ${other.handle} into a low-pressure meetup?`,
          metadata: {
            conversationId,
            messageSampleSize: conversation.messages.length,
          },
        },
      },
      'in_app',
    );

    return { created: true, nudgeId: nudge.id };
  }

  async canSendNudge(userId: string, targetUserId: string | null, type: NudgeType) {
    const now = Date.now();
    const dayAgo = new Date(now - 24 * 60 * 60 * 1000);
    const cooldownStart = new Date(
      now - PAIR_COOLDOWN_HOURS * 60 * 60 * 1000,
    );
    const dismissalStart = new Date(
      now - DISMISSAL_LOOKBACK_DAYS * 24 * 60 * 60 * 1000,
    );

    const [dailyCount, recentPairNudge, recentDismissals] = await Promise.all([
      this.prisma.proactiveNudge.count({
        where: { userId, createdAt: { gte: dayAgo } },
      }),
      targetUserId
        ? this.prisma.proactiveNudge.findFirst({
            where: {
              userId,
              targetUserId,
              type,
              createdAt: { gte: cooldownStart },
            },
            select: { id: true },
          })
        : Promise.resolve(null),
      this.prisma.proactiveNudge.count({
        where: {
          userId,
          accepted: false,
          respondedAt: { gte: dismissalStart },
        },
      }),
    ]);

    return (
      dailyCount < DAILY_NUDGE_CAP &&
      !recentPairNudge &&
      recentDismissals < 3
    );
  }

  async createNudge(
    data: CreateNudgeDto,
    delivery: 'push' | 'in_app' = 'push',
  ) {
    const nudge = await this.prisma.proactiveNudge.create({
      data: {
        userId: data.userId,
        targetUserId: data.context.targetUserId,
        type: data.type,
        payload: data.context as unknown as Prisma.InputJsonValue,
        sentVia: delivery,
        dismissed: false,
      },
    });

    if (delivery === 'push') {
      try {
        await this.notificationsService.sendNudgeNotification(
          data.userId,
          data.type,
          data.context.message || 'Woof has a new suggestion for you.',
          {
            nudgeId: nudge.id,
            ...data.context.metadata,
          },
        );
      } catch (error) {
        this.logger.warn(
          `Push delivery failed for nudge ${nudge.id}: ${this.errorMessage(error)}`,
        );
      }
    }

    return nudge;
  }

  async getUserNudges(userId: string) {
    return this.prisma.proactiveNudge.findMany({
      where: { userId, dismissed: false },
      orderBy: { createdAt: 'desc' },
      take: 10,
      select: {
        id: true,
        targetUserId: true,
        type: true,
        payload: true,
        sentVia: true,
        accepted: true,
        createdAt: true,
      },
    });
  }

  async dismissNudge(nudgeId: string, userId: string) {
    const nudge = await this.findOwnedNudge(nudgeId, userId);
    return this.prisma.proactiveNudge.update({
      where: { id: nudge.id },
      data: {
        dismissed: true,
        accepted: false,
        respondedAt: new Date(),
      },
    });
  }

  async acceptNudge(nudgeId: string, userId: string) {
    const nudge = await this.findOwnedNudge(nudgeId, userId);
    return this.prisma.proactiveNudge.update({
      where: { id: nudge.id },
      data: {
        dismissed: true,
        accepted: true,
        respondedAt: new Date(),
      },
    });
  }

  private async findOwnedNudge(nudgeId: string, userId: string) {
    const nudge = await this.prisma.proactiveNudge.findFirst({
      where: { id: nudgeId, userId },
      select: { id: true },
    });
    if (!nudge) throw new NotFoundException('Nudge not found');
    return nudge;
  }

  private errorMessage(error: unknown) {
    return error instanceof Error ? error.message : String(error);
  }
}
