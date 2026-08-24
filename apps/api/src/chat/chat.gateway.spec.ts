import { ForbiddenException } from '@nestjs/common';
import { MAX_CHAT_MESSAGE_LENGTH } from './chat-input-contract';
import { ChatGateway } from './chat.gateway';

describe('ChatGateway realtime authorization', () => {
  function build() {
    const jwtService = {
      verifyAsync: jest.fn().mockResolvedValue({
        sub: 'user-1',
        exp: Math.floor(Date.now() / 1_000) + 300,
      }),
    };
    const chatSecurity = {
      persistMessage: jest.fn(),
      assertConversationAccess: jest.fn(),
      withAuthorizedRealtimeRecipients: jest.fn(),
    };
    const realtimeAdmission = {
      consume: jest.fn().mockReturnValue({ allowed: true }),
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
      realtimeAdmission as never,
      nudgesService as never
    );
    gateway.server = { to } as never;

    const client = {
      id: 'socket-1',
      handshake: { auth: { token: 'token-1' } },
      join: jest.fn().mockResolvedValue(undefined),
      leave: jest.fn().mockResolvedValue(undefined),
      emit: jest.fn(),
      disconnect: jest.fn(),
    };

    return {
      gateway,
      client,
      jwtService,
      chatSecurity,
      realtimeAdmission,
      nudgesService,
      emit,
      except,
      to,
    };
  }

  it('requires a finite JWT expiry before granting realtime membership', async () => {
    const { gateway, client, jwtService } = build();
    jwtService.verifyAsync.mockResolvedValueOnce({ sub: 'user-1' });

    await gateway.handleConnection(client as never);

    expect(client.join).not.toHaveBeenCalled();
    expect(client.emit).not.toHaveBeenCalled();
    expect(client.disconnect).toHaveBeenCalledTimes(1);
  });

  it('actively expires the socket lease and removes event authority at token expiry', async () => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-08-24T20:30:00.000Z'));

    try {
      const { gateway, client, jwtService, chatSecurity, realtimeAdmission } = build();
      jwtService.verifyAsync.mockResolvedValueOnce({
        sub: 'user-1',
        exp: Math.floor(Date.now() / 1_000) + 2,
      });

      await gateway.handleConnection(client as never);
      expect(client.join).toHaveBeenCalledWith('user:user-1');
      expect(client.disconnect).not.toHaveBeenCalled();

      jest.advanceTimersByTime(1_999);
      expect(client.disconnect).not.toHaveBeenCalled();

      jest.advanceTimersByTime(1);
      expect(client.emit).toHaveBeenCalledWith('session:expired', { reason: 'token_expired' });
      expect(client.disconnect).toHaveBeenCalledTimes(1);

      await expect(
        gateway.handleMessage(client as never, {
          conversationId: 'conversation-1',
          clientMessageId: 'client-message-after-expiry',
          text: 'hello',
        })
      ).resolves.toEqual({ success: false, error: 'unauthorized' });

      expect(realtimeAdmission.consume).not.toHaveBeenCalled();
      expect(chatSecurity.persistMessage).not.toHaveBeenCalled();
    } finally {
      jest.useRealTimers();
    }
  });

  it('rechecks expiry at event ingress even before the disconnect timer gets a turn', async () => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-08-24T20:35:00.000Z'));

    try {
      const { gateway, client, jwtService, chatSecurity, realtimeAdmission } = build();
      jwtService.verifyAsync.mockResolvedValueOnce({
        sub: 'user-1',
        exp: Math.floor(Date.now() / 1_000) + 2,
      });
      await gateway.handleConnection(client as never);

      jest.setSystemTime(new Date('2026-08-24T20:35:03.000Z'));

      await expect(
        gateway.handleTypingStart(client as never, { conversationId: 'conversation-1' })
      ).resolves.toEqual({ success: false, error: 'unauthorized' });

      expect(client.emit).toHaveBeenCalledWith('session:expired', { reason: 'token_expired' });
      expect(client.disconnect).toHaveBeenCalledTimes(1);
      expect(realtimeAdmission.consume).not.toHaveBeenCalled();
      expect(chatSecurity.withAuthorizedRealtimeRecipients).not.toHaveBeenCalled();
    } finally {
      jest.useRealTimers();
    }
  });

  it('cancels the token-expiry timer when a socket disconnects normally', async () => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-08-24T20:40:00.000Z'));

    try {
      const { gateway, client, jwtService } = build();
      jwtService.verifyAsync.mockResolvedValueOnce({
        sub: 'user-1',
        exp: Math.floor(Date.now() / 1_000) + 2,
      });
      await gateway.handleConnection(client as never);

      gateway.handleDisconnect(client as never);
      jest.advanceTimersByTime(3_000);

      expect(client.emit).not.toHaveBeenCalledWith('session:expired', expect.anything());
      expect(client.disconnect).not.toHaveBeenCalled();
    } finally {
      jest.useRealTimers();
    }
  });

  it('delivers persisted messages only through authorized private user rooms', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, to, emit } = build();
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
        text: '  hello  ',
      })
    ).resolves.toMatchObject({ success: true, duplicate: false });

    expect(realtimeAdmission.consume).toHaveBeenCalledWith('user-1', 'message');
    expect(chatSecurity.persistMessage).toHaveBeenCalledWith({
      userId: 'user-1',
      conversationId: 'conversation-1',
      clientMessageId: 'client-message-123',
      text: 'hello',
    });
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

  it('rejects malformed messages before admission or persistence work begins', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission } = build();
    await gateway.handleConnection(client as never);

    const invalidPayloads = [
      null,
      [],
      { conversationId: 'bad id', clientMessageId: 'client-message-123', text: 'hello' },
      { conversationId: 'conversation-1', clientMessageId: 'short', text: 'hello' },
      {
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: 'x'.repeat(MAX_CHAT_MESSAGE_LENGTH + 1),
      },
    ];

    for (const payload of invalidPayloads) {
      await expect(gateway.handleMessage(client as never, payload)).resolves.toEqual({
        success: false,
        error: 'invalid_payload',
      });
    }

    expect(realtimeAdmission.consume).not.toHaveBeenCalled();
    expect(chatSecurity.persistMessage).not.toHaveBeenCalled();
    expect(chatSecurity.withAuthorizedRealtimeRecipients).not.toHaveBeenCalled();
  });

  it('rejects message floods before persistence work begins', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission } = build();
    await gateway.handleConnection(client as never);
    realtimeAdmission.consume.mockReturnValueOnce({ allowed: false, retryAfterMs: 750 });

    await expect(
      gateway.handleMessage(client as never, {
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: 'hello',
      })
    ).resolves.toEqual({ success: false, error: 'rate_limited', retryAfterMs: 750 });

    expect(chatSecurity.persistMessage).not.toHaveBeenCalled();
    expect(chatSecurity.withAuthorizedRealtimeRecipients).not.toHaveBeenCalled();
  });

  it('requires the typing actor to remain authorized and excludes only the current socket', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, to, except, emit } = build();
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

    expect(realtimeAdmission.consume).toHaveBeenCalledWith('user-1', 'typing');
    expect(to).toHaveBeenCalledWith(['user:user-1', 'user:user-2', 'user:user-3']);
    expect(except).toHaveBeenCalledWith('socket-1');
    expect(emit).toHaveBeenCalledWith('typing:start', { userId: 'user-1' });
  });

  it('rejects malformed typing payloads before admission or relationship work', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, to } = build();
    await gateway.handleConnection(client as never);

    await expect(gateway.handleTypingStart(client as never, null)).resolves.toEqual({
      success: false,
      error: 'invalid_payload',
    });

    expect(realtimeAdmission.consume).not.toHaveBeenCalled();
    expect(chatSecurity.withAuthorizedRealtimeRecipients).not.toHaveBeenCalled();
    expect(to).not.toHaveBeenCalled();
  });

  it('rejects typing floods before relationship authorization work begins', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, to } = build();
    await gateway.handleConnection(client as never);
    realtimeAdmission.consume.mockReturnValueOnce({ allowed: false, retryAfterMs: 500 });

    await expect(
      gateway.handleTypingStart(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: false, error: 'rate_limited', retryAfterMs: 500 });

    expect(chatSecurity.withAuthorizedRealtimeRecipients).not.toHaveBeenCalled();
    expect(to).not.toHaveBeenCalled();
  });

  it('rejects malformed membership payloads before admission or access checks', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission } = build();
    await gateway.handleConnection(client as never);
    client.join.mockClear();

    await expect(
      gateway.handleJoinConversation(client as never, { conversationId: 'bad id' })
    ).resolves.toEqual({ success: false, error: 'invalid_payload' });

    expect(realtimeAdmission.consume).not.toHaveBeenCalled();
    expect(chatSecurity.assertConversationAccess).not.toHaveBeenCalled();
    expect(client.join).not.toHaveBeenCalled();
  });

  it('rate-limits room membership churn before access checks', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission } = build();
    await gateway.handleConnection(client as never);
    client.join.mockClear();
    realtimeAdmission.consume.mockReturnValueOnce({ allowed: false, retryAfterMs: 400 });

    await expect(
      gateway.handleJoinConversation(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: false, error: 'rate_limited', retryAfterMs: 400 });

    expect(chatSecurity.assertConversationAccess).not.toHaveBeenCalled();
    expect(client.join).not.toHaveBeenCalled();
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
