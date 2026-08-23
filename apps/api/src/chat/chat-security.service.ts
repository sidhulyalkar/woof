import { BadRequestException, ForbiddenException, Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { acquireRelationshipLocks } from '../trust-safety/relationship-lock';

const MAX_MESSAGE_LENGTH = 4_000;
const CLIENT_MESSAGE_ID_PATTERN = /^[A-Za-z0-9_-]{8,128}$/;

class DuplicateReceiptRaceError extends Error {}

type MessageReceiptRow = {
  message_id: string;
  conversation_id: string;
};

type ConversationAccessClient = Pick<
  Prisma.TransactionClient,
  'conversationParticipant' | 'blockedUser' | '$queryRaw'
>;

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

  async persistMessage(input: {
    userId: string;
    conversationId: string;
    clientMessageId: string;
    text: string;
  }): Promise<{ message: PersistedChatMessage; duplicate: boolean }> {
    await this.assertConversationAccess(input.userId, input.conversationId);

    const text = input.text.trim();
    if (!text) throw new BadRequestException('Message text is required');
    if (text.length > MAX_MESSAGE_LENGTH) {
      throw new BadRequestException(
        `Message text must be ${MAX_MESSAGE_LENGTH} characters or fewer`
      );
    }
    if (!CLIENT_MESSAGE_ID_PATTERN.test(input.clientMessageId)) {
      throw new BadRequestException('clientMessageId is invalid');
    }

    const existing = await this.getReceipt(input.userId, input.clientMessageId);
    if (existing) {
      this.assertReceiptConversation(existing, input.conversationId);
      const message = await this.prisma.message.findUnique({ where: { id: existing.message_id } });
      if (message) return { message, duplicate: true };
    }

    try {
      const message = await this.prisma.$transaction(async (tx) => {
        await this.assertConversationAccessWithClient(
          tx,
          input.userId,
          input.conversationId,
          true
        );

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
