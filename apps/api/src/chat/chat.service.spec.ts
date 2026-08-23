import { ForbiddenException, NotFoundException } from '@nestjs/common';
import { ChatService } from './chat.service';

const directConversation = {
  id: 'conversation-1',
  participants: [{ userId: 'user-1' }, { userId: 'user-2' }],
};

describe('ChatService direct conversation privacy', () => {
  function build() {
    const prisma = {
      user: { findUnique: jest.fn() },
      blockedUser: { findFirst: jest.fn() },
      conversation: { findMany: jest.fn(), create: jest.fn() },
      telemetry: { create: jest.fn() },
    };
    const security = { assertConversationAccess: jest.fn() };
    return {
      prisma,
      service: new ChatService(prisma as never, security as never),
    };
  }

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

  it('creates a conversation only for a public, unblocked profile with no existing direct thread', async () => {
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
    expect(prisma.conversation.create).toHaveBeenCalledTimes(1);
  });
});
