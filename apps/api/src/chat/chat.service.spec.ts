import { ForbiddenException, NotFoundException } from '@nestjs/common';
import { ChatService } from './chat.service';

const directConversation = {
  id: 'conversation-1',
  participants: [{ userId: 'user-1' }, { userId: 'user-2' }],
};

function inboxConversation(input: { id: string; otherUserId: string; name: string }) {
  return {
    id: input.id,
    updatedAt: new Date('2026-08-23T20:00:00.000Z'),
    participants: [
      {
        userId: 'user-1',
        user: { id: 'user-1', handle: 'me', avatarUrl: null, pets: [] },
      },
      {
        userId: input.otherUserId,
        user: {
          id: input.otherUserId,
          handle: input.name,
          avatarUrl: null,
          pets: [{ id: `pet-${input.otherUserId}`, name: `${input.name} dog`, avatarUrl: null }],
        },
      },
    ],
    messages: [
      {
        id: `message-${input.id}`,
        senderId: input.otherUserId,
        text: 'hello',
        mediaUrls: [],
        createdAt: new Date('2026-08-23T19:59:00.000Z'),
      },
    ],
  };
}

describe('ChatService', () => {
  function build() {
    const prisma = {
      user: { findUnique: jest.fn() },
      blockedUser: { findFirst: jest.fn(), findMany: jest.fn() },
      conversation: { findMany: jest.fn(), create: jest.fn() },
      telemetry: { create: jest.fn() },
      $queryRaw: jest.fn().mockResolvedValue([]),
      $transaction: jest.fn(),
    };
    prisma.$transaction.mockImplementation(async (callback: (tx: typeof prisma) => unknown) =>
      callback(prisma)
    );
    const security = { assertConversationAccess: jest.fn() };
    return {
      prisma,
      security,
      service: new ChatService(prisma as never, security as never),
    };
  }

  describe('direct conversation privacy and integrity', () => {
    it('rejects a brand-new conversation with a FRIENDS_ONLY profile', async () => {
      const { prisma, service } = build();
      prisma.user.findUnique.mockResolvedValue({ id: 'user-2', visibility: 'FRIENDS_ONLY' });
      prisma.blockedUser.findFirst.mockResolvedValue(null);
      prisma.conversation.findMany.mockResolvedValue([]);

      await expect(service.createDirectConversation('user-1', 'user-2')).rejects.toBeInstanceOf(
        NotFoundException
      );
      expect(prisma.conversation.create).not.toHaveBeenCalled();
    });

    it('rejects a brand-new conversation with a PRIVATE profile', async () => {
      const { prisma, service } = build();
      prisma.user.findUnique.mockResolvedValue({ id: 'user-2', visibility: 'PRIVATE' });
      prisma.blockedUser.findFirst.mockResolvedValue(null);
      prisma.conversation.findMany.mockResolvedValue([]);

      await expect(service.createDirectConversation('user-1', 'user-2')).rejects.toBeInstanceOf(
        NotFoundException
      );
      expect(prisma.conversation.create).not.toHaveBeenCalled();
    });

    it('reuses an established direct conversation after profile visibility becomes restricted', async () => {
      const { prisma, service } = build();
      prisma.user.findUnique.mockResolvedValue({ id: 'user-2', visibility: 'FRIENDS_ONLY' });
      prisma.blockedUser.findFirst.mockResolvedValue(null);
      prisma.conversation.findMany.mockResolvedValue([directConversation]);

      await expect(service.createDirectConversation('user-1', 'user-2')).resolves.toEqual({
        id: 'conversation-1',
        created: false,
      });
      expect(prisma.conversation.create).not.toHaveBeenCalled();
    });

    it('does not reuse an established conversation when either participant has blocked the other', async () => {
      const { prisma, service } = build();
      prisma.user.findUnique.mockResolvedValue({ id: 'user-2', visibility: 'PUBLIC' });
      prisma.blockedUser.findFirst.mockResolvedValue({ id: 'block-1' });
      prisma.conversation.findMany.mockResolvedValue([directConversation]);

      await expect(service.createDirectConversation('user-1', 'user-2')).rejects.toBeInstanceOf(
        ForbiddenException
      );
      expect(prisma.conversation.create).not.toHaveBeenCalled();
    });

    it('serializes pair lookup and block policy behind transaction-scoped advisory locks', async () => {
      const { prisma, service } = build();
      prisma.user.findUnique.mockResolvedValue({ id: 'user-2', visibility: 'PUBLIC' });
      prisma.blockedUser.findFirst.mockResolvedValue(null);
      prisma.conversation.findMany.mockResolvedValue([]);
      prisma.conversation.create.mockResolvedValue({ id: 'conversation-2' });
      prisma.telemetry.create.mockResolvedValue({ id: 'telemetry-1' });

      await expect(service.createDirectConversation('user-1', 'user-2')).resolves.toEqual({
        id: 'conversation-2',
        created: true,
      });

      expect(prisma.$transaction).toHaveBeenCalledTimes(1);
      expect(prisma.$queryRaw).toHaveBeenCalledTimes(2);
      expect(prisma.$queryRaw.mock.invocationCallOrder[0]).toBeLessThan(
        prisma.conversation.findMany.mock.invocationCallOrder[0]!
      );
      expect(prisma.$queryRaw.mock.invocationCallOrder[1]).toBeLessThan(
        prisma.blockedUser.findFirst.mock.invocationCallOrder[0]!
      );
      expect(prisma.conversation.create).toHaveBeenCalledTimes(1);
      expect(prisma.telemetry.create).toHaveBeenCalledTimes(1);
    });
  });

  describe('inbox batching', () => {
    it('filters blocked direct threads and computes unread counts with bounded set queries', async () => {
      const { prisma, security, service } = build();
      prisma.conversation.findMany.mockResolvedValue([
        inboxConversation({ id: 'conversation-1', otherUserId: 'user-2', name: 'luna-human' }),
        inboxConversation({ id: 'conversation-2', otherUserId: 'user-3', name: 'milo-human' }),
      ]);
      prisma.blockedUser.findMany.mockResolvedValue([{ userId: 'user-3', blockedId: 'user-1' }]);
      prisma.$queryRaw.mockResolvedValue([{ conversation_id: 'conversation-1', unread_count: 3n }]);

      await expect(service.listConversations('user-1')).resolves.toEqual([
        expect.objectContaining({
          id: 'conversation-1',
          participant: expect.objectContaining({ id: 'user-2', name: 'luna-human' }),
          unreadCount: 3,
        }),
      ]);

      expect(prisma.conversation.findMany).toHaveBeenCalledTimes(1);
      expect(prisma.blockedUser.findMany).toHaveBeenCalledTimes(1);
      expect(prisma.$queryRaw).toHaveBeenCalledTimes(1);
      expect(security.assertConversationAccess).not.toHaveBeenCalled();
    });

    it('does not issue block or unread queries when no direct conversations are present', async () => {
      const { prisma, service } = build();
      prisma.conversation.findMany.mockResolvedValue([]);

      await expect(service.listConversations('user-1')).resolves.toEqual([]);
      expect(prisma.blockedUser.findMany).not.toHaveBeenCalled();
      expect(prisma.$queryRaw).not.toHaveBeenCalled();
    });
  });
});
