import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { PrismaService } from '../prisma/prisma.service';
import { NotificationsService } from '../notifications/notifications.service';
import { CreateNudgeDto, NudgeType, NudgeReason } from './dto/create-nudge.dto';

@Injectable()
export class NudgesService {
  private readonly logger = new Logger(NudgesService.name);

  constructor(
    private prisma: PrismaService,
    private notificationsService: NotificationsService,
  ) {}

  /**
   * Check for proximity-based nudge opportunities
   * Runs every 5 minutes
   */
  @Cron(CronExpression.EVERY_5_MINUTES)
  async checkProximityNudges() {
    this.logger.debug('Checking for proximity-based nudges...');

    try {
      // Get recent co-activity segments (last 10 minutes)
      const tenMinutesAgo = new Date(Date.now() - 10 * 60 * 1000);
      const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);

      const recentSegments = await this.prisma.coActivitySegment.findMany({
        where: {
          startTime: {
            gte: tenMinutesAgo,
          },
          endTime: {
            gte: fiveMinutesAgo,
          },
          distanceM: {
            lte: 50, // Within 50 meters
          },
          otherUserId: {
            not: null,
          },
        },
      });

      this.logger.debug(`Found ${recentSegments.length} recent co-activity segments`);

      for (const segment of recentSegments) {
        if (!segment.otherUserId) continue;

        // Check if users are compatible
        const compatibility = await this.checkCompatibility(
          segment.userId,
          segment.otherUserId,
        );

        if (compatibility && compatibility.compatibilityScore && compatibility.compatibilityScore >= 0.7) {
          // Check cooldown
          const canSend = await this.canSendNudge(
            segment.userId,
            segment.otherUserId,
            NudgeType.MEETUP,
          );

          if (canSend) {
            // Create nudges for both users
            await this.createProximityNudge(segment);
            this.logger.log(
              `Created proximity nudge for users ${segment.userId} and ${segment.otherUserId}`,
            );
          }
        }
      }
    } catch (error) {
      this.logger.error(`Error checking proximity nudges: ${error.message}`, error.stack);
    }
  }

  /**
   * Check for chat activity-based nudges
   * Triggered by chat service when message count reaches threshold
   */
  async checkChatActivityNudges(conversationId: string) {
    this.logger.debug(`Checking chat activity nudge for conversation ${conversationId}`);

    try {
      // Get conversation details
      const conversation = await this.prisma.conversation.findUnique({
        where: { id: conversationId },
        include: {
          participants: {
            include: {
              user: true,
            },
          },
          messages: {
            orderBy: { createdAt: 'desc' },
            take: 10,
          },
        },
      });

      if (!conversation || conversation.participants.length !== 2) {
        return;
      }

      const messageCount = conversation.messages.length;

      // If conversation has 5+ messages, suggest meetup
      if (messageCount >= 5) {
        const [user1, user2] = conversation.participants.map((p: any) => p.user);

        // Check cooldown
        const canSend = await this.canSendNudge(user1.id, user2.id, NudgeType.MEETUP);

        if (canSend) {
          // Create nudge for both users
          await Promise.all([
            this.createNudge({
              userId: user1.id,
              type: NudgeType.MEETUP,
              context: {
                targetUserId: user2.id,
                reason: NudgeReason.CHAT_ACTIVITY,
                message: `You and ${user2.handle} have been chatting – ready to meet up?`,
                metadata: {
                  conversationId,
                  messageCount,
                },
              },
            }),
            this.createNudge({
              userId: user2.id,
              type: NudgeType.MEETUP,
              context: {
                targetUserId: user1.id,
                reason: NudgeReason.CHAT_ACTIVITY,
                message: `You and ${user1.handle} have been chatting – ready to meet up?`,
                metadata: {
                  conversationId,
                  messageCount,
                },
              },
            }),
          ]);

          this.logger.log(
            `Created chat activity nudges for users ${user1.id} and ${user2.id}`,
          );
        }
      }
    } catch (error) {
      this.logger.error(
        `Error checking chat activity nudges: ${error.message}`,
        error.stack,
      );
    }
  }

  /**
   * Create a proximity-based nudge for both users
   */
  private async createProximityNudge(segment: any) {
    const { userId, otherUserId, distanceM, venueType } = segment;

    // Get user details for personalized messages
    const [user1, user2] = await Promise.all([
      this.prisma.user.findUnique({
        where: { id: userId },
        include: { pets: { take: 1 } }
      }),
      this.prisma.user.findUnique({
        where: { id: otherUserId },
        include: { pets: { take: 1 } }
      }),
    ]);

    if (!user1 || !user2) return;

    await Promise.all([
      this.createNudge({
        userId: user1.id,
        type: NudgeType.MEETUP,
        context: {
          targetUserId: user2.id,
          reason: NudgeReason.PROXIMITY,
          message: `${user2.handle} is nearby! Want to meet up?`,
          metadata: {
            distance: Math.round(distanceM),
            venueType,
            petNames: {
              yours: user1.pets[0]?.name,
              theirs: user2.pets[0]?.name,
            },
          },
        },
      }),
      this.createNudge({
        userId: user2.id,
        type: NudgeType.MEETUP,
        context: {
          targetUserId: user1.id,
          reason: NudgeReason.PROXIMITY,
          message: `${user1.handle} is nearby! Want to meet up?`,
          metadata: {
            distance: Math.round(distanceM),
            venueType,
            petNames: {
              yours: user2.pets[0]?.name,
              theirs: user1.pets[0]?.name,
            },
          },
        },
      }),
    ]);
  }

  /**
   * Check if a nudge can be sent (cooldown enforcement)
   */
  async canSendNudge(
    userId1: string,
    userId2: string,
    type: NudgeType,
  ): Promise<boolean> {
    const cooldownHours = 24;
    const cooldownTime = new Date(Date.now() - cooldownHours * 60 * 60 * 1000);

    // Check if a similar nudge was sent recently
    const recentNudge = await this.prisma.proactiveNudge.findFirst({
      where: {
        userId: userId1,
        type,
        targetUserId: userId2,
        createdAt: {
          gte: cooldownTime,
        },
      },
    });

    return !recentNudge;
  }

  /**
   * Create a nudge record and send push notification
   */
  async createNudge(data: CreateNudgeDto) {
    const nudge = await this.prisma.proactiveNudge.create({
      data: {
        userId: data.userId,
        type: data.type,
        payload: data.context as any,
        dismissed: false,
      },
    });

    // Send push notification for the nudge
    try {
      await this.notificationsService.sendNudgeNotification(
        data.userId,
        data.type,
        data.context.message || 'New suggestion for you!',
        {
          nudgeId: nudge.id,
          ...data.context.metadata,
        },
      );
    } catch (error) {
      this.logger.error(
        `Failed to send push notification for nudge: ${error.message}`,
      );
      // Don't fail nudge creation if push fails
    }

    return nudge;
  }

  /**
   * Get active nudges for a user
   */
  async getUserNudges(userId: string) {
    return this.prisma.proactiveNudge.findMany({
      where: {
        userId,
        dismissed: false,
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
  }

  /**
   * Dismiss a nudge
   */
  async dismissNudge(nudgeId: string, userId: string) {
    return this.prisma.proactiveNudge.update({
      where: {
        id: nudgeId,
        userId, // Ensure user owns this nudge
      },
      data: {
        dismissed: true,
      },
    });
  }

  /**
   * Accept a nudge (mark as interacted)
   */
  async acceptNudge(nudgeId: string, userId: string) {
    const nudge = await this.prisma.proactiveNudge.update({
      where: {
        id: nudgeId,
        userId,
      },
      data: {
        dismissed: true, // Remove from active list
      },
    });

    // Return the nudge data so caller can take action
    return nudge;
  }

  /**
   * Check compatibility between two users
   */
  private async checkCompatibility(userId1: string, userId2: string) {
    // Get or calculate compatibility score
    const compatibility = await this.prisma.petEdge.findFirst({
      where: {
        OR: [
          { petA: { ownerId: userId1 }, petB: { ownerId: userId2 } },
          { petA: { ownerId: userId2 }, petB: { ownerId: userId1 } },
        ],
      },
    });

    return compatibility;
  }
}
