import { BadRequestException, ForbiddenException, Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import {
  acquireParticipantRelationshipLocks,
  acquireRelationshipLocks,
} from '../trust-safety/relationship-lock';
import {
  isChatIdentifier,
  isClientMessageId,
  MAX_CHAT_MESSAGE_LENGTH,
  normalizeChatMessageText,
} from './chat-input-contract';

class DuplicateReceiptRaceError extends Error {}

type MessageReceiptRow = {
  message_id: string;
  conversation_id: string;
};

type ConversationAccessClient = Pick<
  Prisma.TransactionClient,
  'conversationParticipant' | 'blockedUser' | '$queryRaw'
>;

type RealtimeDelivery = (authorizedUserIds: string[]) => void | Promise<void>;

export type PersistedChatMessage = {
  id: string;
  conversationId: string;
  senderId: string;
  text: string;
  mediaUrls: string[];
  createdAt: Date;
};

@Injectable()
export class ChatSecurityService {
  constructor(private readonly prisma: PrismaService) {}

  async assertConversationAccess(userId: string, conversationId: string) {
    return this.assertConversationAccessWithClient(this.prisma, userId, conversationId, false);
  }

  async withAuthorizedRealtimeRecipients(
    conversationId: string,
    deliver: RealtimeDelivery,
    requiredUserId?: string
  ) {
    return this.prisma.$transaction(async (tx) => {
      const participants = await tx.conversationParticipant.findMany({
        where: { conversationId },
        select: { userId: true },
        orderBy: { userId: 'asc' },
      });
      const participantUserIds = [...new Set(participants.map((entry) => entry.userId))];

      if (participantUserIds.length < 2) {
        throw new ForbiddenException('Conversation is unavailable');
      }

      await acquireParticipantRelationshipLocks(tx, participantUserIds);

      const blocks = await tx.blockedUser.findMany({
        where: {
          userId: { in: participantUserIds },
          blockedId: { in: participantUserIds },
        },
        select: { userId: true, blockedId: true },
      });

      const blockedEndpoints = new Set<string>();
      for (const block of blocks) {
        if (block.userId === block.blockedId) continue;
        blockedEndpoints.add(block.userId);
        blockedEndpoints.add(block.blockedId);
      }

      const authorizedUserIds = participantUserIds.filter(
        (participantUserId) => !blockedEndpoints.has(participantUserId)
      );

      if (requiredUserId && !authorizedUserIds.includes(requiredUserId)) {
        throw new ForbiddenException('Conversation is unavailable');
      }

      if (authorizedUserIds.length > 0) {
        await deliver(authorizedUserIds);
      }

      return { authorizedUserIds };
    });
  }

  async persistMessage(input: {
    userId: string;
    conversationId: string;
    clientMessageId: string;
    text: string;
  }): Promise<{ message: PersistedChatMessage; duplicate: boolean }> {
    if (!isChatIdentifier(input.conversationId)) {
      throw new BadRequestException('conversationId is invalid');
    }
    if (!isClientMessageId(input.clientMessageId)) {
      throw new BadRequestException('clientMessageId is invalid');
    }
    if (typeof input.text !== 'string') {
      throw new BadRequestException('Message text is required');
    }
    if (input.text.length > MAX_CHAT_MESSAGE_LENGTH) {
      throw new BadRequestException(
        `Message text must be ${MAX_CHAT_MESSAGE_LENGTH} characters or fewer`
      );
    }
    if (input.text.includes('\u0000')) {
      throw new BadRequestException('Message text is invalid');
    }

    const text = normalizeChatMessageText(input.text);
    if (!text) throw new BadRequestException('Message text is required');

    await this.assertConversationAccess(input.userId, input.conversationId);

    const existing = await this.getReceipt(input.userId, input.clientMessageId);
    if (existing) {
      this.assertReceiptConversation(existing, input.conversationId);
      const message = await this.prisma.message.findUnique({ where: { id: existing.message_id } });
      if (message) return { message, duplicate: true };
    }

    try {
      const message = await this.prisma.$transaction(async (tx) => {
        await this.assertConversationAccessWithClient(tx, input.userId, input.conversationId, true);

        const receiptRows = await tx.$queryRaw<MessageReceiptRow[]>(Prisma.sql`
          SELECT message_id, conversation_id
          FROM dogos_chat.message_receipts
          WHERE user_id = CAST(${input.userId} AS text)
            AND client_message_id = ${input.clientMessageId}
          LIMIT 1
        `);
        if (receiptRows[0]) {
          this.assertReceiptConversation(receiptRows[0], input.conversationId);
          const persisted = await tx.message.findUnique({
            where: { id: receiptRows[0].message_id },
          });
          if (!persisted) throw new DuplicateReceiptRaceError();
          return { message: persisted, duplicate: true };
        }

        const persisted = await tx.message.create({
          data: {
            conversationId: input.conversationId,
            senderId: input.userId,
            text,
          },
        });

        const inserted = await tx.$queryRaw<Array<{ message_id: string }>>(Prisma.sql`
          INSERT INTO dogos_chat.message_receipts (
            user_id, client_message_id, conversation_id, message_id
          ) VALUES (
            CAST(${input.userId} AS text),
            ${input.clientMessageId},
            CAST(${input.conversationId} AS text),
            CAST(${persisted.id} AS text)
          )
          ON CONFLICT (user_id, client_message_id) DO NOTHING
          RETURNING message_id
        `);

        if (inserted.length === 0) throw new DuplicateReceiptRaceError();
        return { message: persisted, duplicate: false };
      });

      return message;
    } catch (error) {
      if (!(error instanceof DuplicateReceiptRaceError)) throw error;
      const raced = await this.getReceipt(input.userId, input.clientMessageId);
      if (!raced) throw error;
      this.assertReceiptConversation(raced, input.conversationId);
      const message = await this.prisma.message.findUnique({ where: { id: raced.message_id } });
      if (!message) throw error;
      return { message, duplicate: true };
    }
  }

  private async assertConversationAccessWithClient(
    client: ConversationAccessClient,
    userId: string,
    conversationId: string,
    lockRelationships: boolean
  ) {
    const participant = await client.conversationParticipant.findUnique({
      where: { conversationId_userId: { conversationId, userId } },
      select: {
        conversation: {
          select: {
            participants: { select: { userId: true } },
          },
        },
      },
    });

    if (!participant) {
      throw new ForbiddenException('Conversation is unavailable');
    }

    const otherUserIds = participant.conversation.participants
      .map((entry) => entry.userId)
      .filter((participantUserId) => participantUserId !== userId);

    if (otherUserIds.length === 0) {
      throw new ForbiddenException('Conversation is unavailable');
    }

    if (lockRelationships) {
      await acquireRelationshipLocks(client, userId, otherUserIds);
    }

    const blocked = await client.blockedUser.findFirst({
      where: {
        OR: otherUserIds.flatMap((otherUserId) => [
          { userId, blockedId: otherUserId },
          { userId: otherUserId, blockedId: userId },
        ]),
      },
      select: { id: true },
    });

    if (blocked) {
      throw new ForbiddenException('Conversation is unavailable');
    }

    return { otherUserIds };
  }

  private assertReceiptConversation(receipt: MessageReceiptRow, conversationId: string) {
    if (receipt.conversation_id !== conversationId) {
      throw new BadRequestException('clientMessageId has already been used');
    }
  }

  private async getReceipt(userId: string, clientMessageId: string) {
    const rows = await this.prisma.$queryRaw<MessageReceiptRow[]>(Prisma.sql`
      SELECT message_id, conversation_id
      FROM dogos_chat.message_receipts
      WHERE user_id = CAST(${userId} AS text)
        AND client_message_id = ${clientMessageId}
      LIMIT 1
    `);
    return rows[0] ?? null;
  }
}
