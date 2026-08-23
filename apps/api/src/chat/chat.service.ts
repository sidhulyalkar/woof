import {
  BadRequestException,
  ForbiddenException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { ChatSecurityService } from './chat-security.service';

const MAX_CONVERSATIONS = 50;
const MAX_MESSAGES = 100;

type UnreadCountRow = {
  conversation_id: string;
  unread_count: bigint | number;
};

function directPairLockKey(userId: string, participantId: string) {
  return `woof:direct-chat:${[userId, participantId].sort().join(':')}`;
}

@Injectable()
export class ChatService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly security: ChatSecurityService
  ) {}

  async listConversations(userId: string) {
    const conversations = await this.prisma.conversation.findMany({
      where: { participants: { some: { userId } } },
      orderBy: { updatedAt: 'desc' },
      take: MAX_CONVERSATIONS,
      select: {
        id: true,
        updatedAt: true,
        participants: {
          select: {
            userId: true,
            user: {
              select: {
                id: true,
                handle: true,
                avatarUrl: true,
                pets: {
                  orderBy: { createdAt: 'asc' },
                  take: 1,
                  select: { id: true, name: true, avatarUrl: true },
                },
              },
            },
          },
        },
        messages: {
          orderBy: { createdAt: 'desc' },
          take: 1,
          select: {
            id: true,
            senderId: true,
            text: true,
            mediaUrls: true,
            createdAt: true,
          },
        },
      },
    });

    const directConversations = conversations.filter((conversation) => {
      if (conversation.participants.length !== 2) return false;
      return conversation.participants.some((participant) => participant.userId === userId);
    });
    if (directConversations.length === 0) return [];

    const otherUserIds = [
      ...new Set(
        directConversations.flatMap((conversation) =>
          conversation.participants
            .filter((participant) => participant.userId !== userId)
            .map((participant) => participant.userId)
        )
      ),
    ];

    const blockedRelations =
      otherUserIds.length === 0
        ? []
        : await this.prisma.blockedUser.findMany({
            where: {
              OR: [
                { userId, blockedId: { in: otherUserIds } },
                { userId: { in: otherUserIds }, blockedId: userId },
              ],
            },
            select: { userId: true, blockedId: true },
          });
    const blockedOtherUserIds = new Set(
      blockedRelations.map((relation) =>
        relation.userId === userId ? relation.blockedId : relation.userId
      )
    );

    const visible = directConversations.filter((conversation) => {
      const other = conversation.participants.find((participant) => participant.userId !== userId);
      return Boolean(other && !blockedOtherUserIds.has(other.userId));
    });
    if (visible.length === 0) return [];

    const unreadCounts = await this.getUnreadCounts(
      userId,
      visible.map((conversation) => conversation.id)
    );

    return visible.map((conversation) => {
      const other = conversation.participants.find((participant) => participant.userId !== userId)!;
      const lastMessage = conversation.messages[0] ?? null;

      return {
        id: conversation.id,
        participant: {
          id: other.user.id,
          name: other.user.handle,
          avatarUrl: other.user.avatarUrl,
          petId: other.user.pets[0]?.id ?? null,
          petName: other.user.pets[0]?.name ?? null,
          petAvatarUrl: other.user.pets[0]?.avatarUrl ?? null,
        },
        lastMessage: lastMessage
          ? {
              id: lastMessage.id,
              senderId: lastMessage.senderId,
              content: lastMessage.text,
              mediaUrls: lastMessage.mediaUrls,
              createdAt: lastMessage.createdAt,
            }
          : null,
        unreadCount: unreadCounts.get(conversation.id) ?? 0,
        updatedAt: conversation.updatedAt,
      };
    });
  }

  async createDirectConversation(userId: string, participantId: string) {
    if (!participantId || participantId === userId) {
      throw new BadRequestException('Choose another member');
    }

    const pairKey = directPairLockKey(userId, participantId);
    return this.prisma.$transaction(async (tx) => {
      await tx.$queryRaw(Prisma.sql`
        SELECT 1 AS locked
        FROM (
          SELECT pg_advisory_xact_lock(hashtextextended(${pairKey}, 0))
        ) AS pair_lock
      `);

      const [target, blocked, candidates] = await Promise.all([
        tx.user.findUnique({
          where: { id: participantId },
          select: { id: true, visibility: true },
        }),
        tx.blockedUser.findFirst({
          where: {
            OR: [
              { userId, blockedId: participantId },
              { userId: participantId, blockedId: userId },
            ],
          },
          select: { id: true },
        }),
        tx.conversation.findMany({
          where: {
            AND: [
              { participants: { some: { userId } } },
              { participants: { some: { userId: participantId } } },
            ],
          },
          select: { id: true, participants: { select: { userId: true } } },
          take: 20,
        }),
      ]);

      if (blocked) throw new ForbiddenException('Conversation is unavailable');

      const existing = candidates.find(
        (conversation) =>
          conversation.participants.length === 2 &&
          conversation.participants.some((participant) => participant.userId === userId) &&
          conversation.participants.some((participant) => participant.userId === participantId)
      );
      if (existing) return { id: existing.id, created: false };

      if (!target || target.visibility !== 'PUBLIC') {
        throw new NotFoundException('Member not found');
      }

      const conversation = await tx.conversation.create({
        data: {
          participants: {
            create: [{ userId }, { userId: participantId }],
          },
        },
        select: { id: true },
      });
      await tx.telemetry.create({
        data: {
          userId,
          source: 'chat',
          event: 'CONVERSATION_STARTED',
          data: { conversationId: conversation.id },
        },
      });
      return { id: conversation.id, created: true };
    });
  }

  async getMessages(userId: string, conversationId: string, page = 1, limit = 50) {
    await this.security.assertConversationAccess(userId, conversationId);
    const safePage = Math.max(1, Number(page) || 1);
    const safeLimit = Math.max(1, Math.min(Number(limit) || 50, MAX_MESSAGES));
    const skip = (safePage - 1) * safeLimit;
    const [messages, total] = await Promise.all([
      this.prisma.message.findMany({
        where: { conversationId },
        orderBy: { createdAt: 'desc' },
        skip,
        take: safeLimit,
        select: {
          id: true,
          conversationId: true,
          senderId: true,
          text: true,
          mediaUrls: true,
          createdAt: true,
        },
      }),
      this.prisma.message.count({ where: { conversationId } }),
    ]);

    return {
      data: messages.reverse().map((message) => ({
        id: message.id,
        conversationId: message.conversationId,
        senderId: message.senderId,
        content: message.text,
        type: message.mediaUrls.length > 0 ? 'image' : 'text',
        mediaUrl: message.mediaUrls[0] ?? null,
        createdAt: message.createdAt,
      })),
      total,
      page: safePage,
      limit: safeLimit,
    };
  }

  async markRead(userId: string, conversationId: string) {
    await this.security.assertConversationAccess(userId, conversationId);
    await this.prisma.conversationParticipant.update({
      where: { conversationId_userId: { conversationId, userId } },
      data: { lastReadAt: new Date() },
    });
    return { ok: true };
  }

  private async getUnreadCounts(userId: string, conversationIds: string[]) {
    if (conversationIds.length === 0) return new Map<string, number>();

    const rows = await this.prisma.$queryRaw<UnreadCountRow[]>(Prisma.sql`
      SELECT
        cp.conversation_id,
        COUNT(m.id) AS unread_count
      FROM conversation_participants cp
      LEFT JOIN messages m
        ON m.conversation_id = cp.conversation_id
       AND m.sender_id <> ${userId}
       AND (cp.last_read_at IS NULL OR m.created_at > cp.last_read_at)
      WHERE cp.user_id = ${userId}
        AND cp.conversation_id IN (${Prisma.join(conversationIds)})
      GROUP BY cp.conversation_id
    `);

    return new Map(rows.map((row) => [row.conversation_id, Number(row.unread_count)]));
  }
}
