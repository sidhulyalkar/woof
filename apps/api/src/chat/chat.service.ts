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
            lastReadAt: true,
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

    const visible = [];
    for (const conversation of conversations) {
      try {
        await this.security.assertConversationAccess(userId, conversation.id);
      } catch {
        continue;
      }

      const self = conversation.participants.find((participant) => participant.userId === userId);
      const others = conversation.participants.filter(
        (participant) => participant.userId !== userId
      );
      if (others.length !== 1) continue;
      const other = others[0]!;
      const lastMessage = conversation.messages[0] ?? null;
      const unreadCount = await this.prisma.message.count({
        where: {
          conversationId: conversation.id,
          senderId: { not: userId },
          ...(self?.lastReadAt ? { createdAt: { gt: self.lastReadAt } } : {}),
        },
      });

      visible.push({
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
        unreadCount,
        updatedAt: conversation.updatedAt,
      });
    }

    return visible;
  }

  async createDirectConversation(userId: string, participantId: string) {
    if (!participantId || participantId === userId) {
      throw new BadRequestException('Choose another member');
    }

    const [target, blocked, candidates] = await Promise.all([
      this.prisma.user.findUnique({
        where: { id: participantId },
        select: { id: true, visibility: true },
      }),
      this.prisma.blockedUser.findFirst({
        where: {
          OR: [
            { userId, blockedId: participantId },
            { userId: participantId, blockedId: userId },
          ],
        },
        select: { id: true },
      }),
      this.prisma.conversation.findMany({
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

    const conversation = await this.prisma.conversation.create({
      data: {
        participants: {
          create: [{ userId }, { userId: participantId }],
        },
      },
      select: { id: true },
    });
    await this.recordTelemetry(userId, 'CONVERSATION_STARTED', { conversationId: conversation.id });
    return { id: conversation.id, created: true };
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

  private async recordTelemetry(userId: string, event: string, data: Prisma.InputJsonObject) {
    await this.prisma.telemetry.create({
      data: { userId, source: 'chat', event, data },
    });
  }
}
