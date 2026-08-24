import { ForbiddenException } from '@nestjs/common';
import { ChatGateway } from './chat.gateway';

describe('ChatGateway realtime authorization', () => {
  function build() {
    const jwtService = {
      verifyAsync: jest.fn().mockResolvedValue({ sub: 'user-1' }),
    };
    const chatSecurity = {
      persistMessage: jest.fn(),
      assertConversationAccess: jest.fn(),
      withAuthorizedRealtimeRecipients: jest.fn(),
    };
    const nudgesService = {
      checkChatActivityNudges: jest.fn().mockResolvedValue({ created: false }),
    };
    const emit = jest.fn();
    const except = jest.fn().mockReturnValue({ emit });
    const to = jest.fn().mockReturnValue({ emit, except });
    const gateway = new ChatGateway(
      jwtService as never,
      chatSecurity as never,
      nudgesService as never
    );
    gateway.server = { to } as never;

    const client = {
      id: 'socket-1',
      handshake: { auth: { token: 'token-1' } },
      join: jest.fn().mockResolvedValue(undefined),
      leave: jest.fn().mockResolvedValue(undefined),
      disconnect: jest.fn(),
    };

    return {
      gateway,
      client,
      jwtService,
      chatSecurity,
      nudgesService,
      emit,
      except,
      to,
    };
  }

  it('delivers persisted messages only through authorized private user rooms', async () => {
    const { gateway, client, chatSecurity, to, emit } = build();
    await gateway.handleConnection(client as never);

    const createdAt = new Date('2026-08-24T04:00:00.000Z');
    chatSecurity.persistMessage.mockResolvedValue({
      duplicate: false,
      message: {
        id: 'message-1',
        conversationId: 'conversation-1',
        senderId: 'user-1',
        text: 'hello',
        mediaUrls: [],
        createdAt,
      },
    });
    chatSecurity.withAuthorizedRealtimeRecipients.mockImplementation(
      async (_conversationId: string, deliver: (userIds: string[]) => void) => {
        deliver(['user-1', 'user-3']);
        return { authorizedUserIds: ['user-1', 'user-3'] };
      }
    );

    await expect(
      gateway.handleMessage(client as never, {
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: 'hello',
      })
    ).resolves.toMatchObject({ success: true, duplicate: false });

    expect(chatSecurity.withAuthorizedRealtimeRecipients).toHaveBeenCalledWith(
      'conversation-1',
      expect.any(Function)
    );
    expect(to).toHaveBeenCalledWith(['user:user-1', 'user:user-3']);
    expect(emit).toHaveBeenCalledWith('message:received', {
      id: 'message-1',
      conversationId: 'conversation-1',
      senderId: 'user-1',
      text: 'hello',
      mediaUrls: [],
      timestamp: createdAt,
    });
    expect(
      to.mock.calls.some(([rooms]) =>
        Array.isArray(rooms)
          ? rooms.some((room) => String(room).startsWith('conversation:'))
          : String(rooms).startsWith('conversation:')
      )
    ).toBe(false);
  });

  it('requires the typing actor to remain authorized and excludes only the current socket', async () => {
    const { gateway, client, chatSecurity, to, except, emit } = build();
    await gateway.handleConnection(client as never);

    chatSecurity.withAuthorizedRealtimeRecipients.mockImplementation(
      async (
        _conversationId: string,
        deliver: (userIds: string[]) => void,
        requiredUserId?: string
      ) => {
        expect(requiredUserId).toBe('user-1');
        deliver(['user-1', 'user-2', 'user-3']);
        return { authorizedUserIds: ['user-1', 'user-2', 'user-3'] };
      }
    );

    await expect(
      gateway.handleTypingStart(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: true });

    expect(to).toHaveBeenCalledWith(['user:user-1', 'user:user-2', 'user:user-3']);
    expect(except).toHaveBeenCalledWith('socket-1');
    expect(emit).toHaveBeenCalledWith('typing:start', { userId: 'user-1' });
  });

  it('does not emit typing after relationship authorization is revoked', async () => {
    const { gateway, client, chatSecurity, to } = build();
    await gateway.handleConnection(client as never);
    chatSecurity.withAuthorizedRealtimeRecipients.mockRejectedValue(
      new ForbiddenException('Conversation is unavailable')
    );

    await expect(
      gateway.handleTypingStop(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: false, error: 'conversation_unavailable' });

    expect(to).not.toHaveBeenCalled();
  });
});
