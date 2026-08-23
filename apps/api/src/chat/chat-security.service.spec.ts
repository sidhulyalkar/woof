import { BadRequestException, ForbiddenException } from '@nestjs/common';
import { ChatSecurityService } from './chat-security.service';

describe('ChatSecurityService', () => {
  const participant = (userIds = ['user-1', 'user-2']) => ({
    conversation: {
      participants: userIds.map((userId) => ({ userId })),
    },
  });

  function build() {
    const prisma = {
      conversationParticipant: { findUnique: jest.fn() },
      blockedUser: { findFirst: jest.fn() },
      message: { findUnique: jest.fn() },
      $queryRaw: jest.fn(),
      $transaction: jest.fn(),
    };
    return {
      prisma,
      service: new ChatSecurityService(prisma as never),
    };
  }

  it('rejects a user who is not a conversation participant', async () => {
    const { prisma, service } = build();
    prisma.conversationParticipant.findUnique.mockResolvedValue(null);

    await expect(service.assertConversationAccess('user-1', 'conversation-1')).rejects.toBeInstanceOf(
      ForbiddenException,
    );
    expect(prisma.blockedUser.findFirst).not.toHaveBeenCalled();
  });

  it('rejects access when either participant has blocked the other', async () => {
    const { prisma, service } = build();
    prisma.conversationParticipant.findUnique.mockResolvedValue(participant());
    prisma.blockedUser.findFirst.mockResolvedValue({ id: 'block-1' });

    await expect(service.assertConversationAccess('user-1', 'conversation-1')).rejects.toBeInstanceOf(
      ForbiddenException,
    );
  });

  it('allows an unblocked participant and returns only the other participant ids', async () => {
    const { prisma, service } = build();
    prisma.conversationParticipant.findUnique.mockResolvedValue(participant(['user-1', 'user-2', 'user-3']));
    prisma.blockedUser.findFirst.mockResolvedValue(null);

    await expect(service.assertConversationAccess('user-1', 'conversation-1')).resolves.toEqual({
      otherUserIds: ['user-2', 'user-3'],
    });
  });

  it('replays an idempotent persisted message without creating or emitting a second record', async () => {
    const { prisma, service } = build();
    prisma.conversationParticipant.findUnique.mockResolvedValue(participant());
    prisma.blockedUser.findFirst.mockResolvedValue(null);
    prisma.$queryRaw.mockResolvedValue([
      { message_id: 'message-1', conversation_id: 'conversation-1' },
    ]);
    const persisted = {
      id: 'message-1',
      conversationId: 'conversation-1',
      senderId: 'user-1',
      text: 'hello',
      mediaUrls: [],
      createdAt: new Date('2026-08-23T08:00:00.000Z'),
    };
    prisma.message.findUnique.mockResolvedValue(persisted);

    await expect(
      service.persistMessage({
        userId: 'user-1',
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: 'hello',
      }),
    ).resolves.toEqual({ message: persisted, duplicate: true });
    expect(prisma.$transaction).not.toHaveBeenCalled();
  });

  it('rejects empty text before any message persistence transaction begins', async () => {
    const { prisma, service } = build();
    prisma.conversationParticipant.findUnique.mockResolvedValue(participant());
    prisma.blockedUser.findFirst.mockResolvedValue(null);

    await expect(
      service.persistMessage({
        userId: 'user-1',
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: '   ',
      }),
    ).rejects.toBeInstanceOf(BadRequestException);
    expect(prisma.$transaction).not.toHaveBeenCalled();
  });
});
