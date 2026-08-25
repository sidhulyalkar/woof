import { ForbiddenException } from '@nestjs/common';
import { MAX_CHAT_MESSAGE_LENGTH } from './chat-input-contract';
import { ChatGateway } from './chat.gateway';

describe('ChatGateway realtime authorization', () => {
  function build() {
    const authorityTx = { kind: 'authority-transaction' };
    const jwtService = {
      verifyAsync: jest.fn().mockResolvedValue({
        sub: 'user-1',
        sid: 'session-1',
        exp: Math.floor(Date.now() / 1_000) + 300,
      }),
    };
    const sessionAuthority = {
      withActiveSession: jest
        .fn()
        .mockImplementation(
          async (
            _sessionId: string,
            _userId: string,
            work: (tx: unknown) => unknown | Promise<unknown>
          ) => ({
            authorized: true,
            result: await work(authorityTx),
          })
        ),
      withActiveSessions: jest
        .fn()
        .mockImplementation(
          async (
            sessionIds: string[],
            deliver: (activeSessionIds: Set<string>) => void | Promise<void>
          ) => {
            const activeSessionIds = new Set(sessionIds);
            await deliver(activeSessionIds);
            return { activeSessionIds };
          }
        ),
      withActiveSessionsInTransaction: jest
        .fn()
        .mockImplementation(
          async (
            tx: unknown,
            sessionIds: string[],
            deliver: (activeSessionIds: Set<string>) => void | Promise<void>
          ) => {
            expect(tx).toBe(authorityTx);
            const activeSessionIds = new Set(sessionIds);
            await deliver(activeSessionIds);
            return { activeSessionIds };
          }
        ),
    };
    const chatSecurity = {
      persistMessage: jest.fn(),
      persistMessageInTransaction: jest.fn(),
      assertConversationAccess: jest.fn(),
      assertConversationAccessInTransaction: jest.fn(),
      withAuthorizedRealtimeRecipients: jest.fn(),
      withAuthorizedRealtimeRecipientsInTransaction: jest.fn(),
    };
    const realtimeAdmission = {
      consume: jest.fn().mockReturnValue({ allowed: true }),
    };
    const nudgesService = {
      checkChatActivityNudges: jest.fn().mockResolvedValue({ created: false }),
    };
    const emit = jest.fn();
    const operator = { emit, except: jest.fn() };
    operator.except.mockReturnValue(operator);
    const to = jest.fn().mockReturnValue(operator);
    const gateway = new ChatGateway(
      jwtService as never,
      sessionAuthority as never,
      chatSecurity as never,
      realtimeAdmission as never,
      nudgesService as never
    );
    gateway.server = { to } as never;

    const makeClient = (id = 'socket-1') => ({
      id,
      handshake: { auth: { token: `token-${id}` } },
      join: jest.fn().mockResolvedValue(undefined),
      leave: jest.fn().mockResolvedValue(undefined),
      emit: jest.fn(),
      disconnect: jest.fn(),
    });
    const client = makeClient();

    return {
      gateway,
      client,
      makeClient,
      authorityTx,
      jwtService,
      sessionAuthority,
      chatSecurity,
      realtimeAdmission,
      nudgesService,
      emit,
      except: operator.except,
      to,
    };
  }

  it('requires a finite JWT expiry before granting realtime membership', async () => {
    const { gateway, client, jwtService, sessionAuthority } = build();
    jwtService.verifyAsync.mockResolvedValueOnce({ sub: 'user-1', sid: 'session-1' });

    await gateway.handleConnection(client as never);

    expect(sessionAuthority.withActiveSession).not.toHaveBeenCalled();
    expect(client.join).not.toHaveBeenCalled();
    expect(client.emit).not.toHaveBeenCalled();
    expect(client.disconnect).toHaveBeenCalledTimes(1);
  });

  it('requires a persisted session id before granting realtime membership', async () => {
    const { gateway, client, jwtService, sessionAuthority } = build();
    jwtService.verifyAsync.mockResolvedValueOnce({
      sub: 'user-1',
      exp: Math.floor(Date.now() / 1_000) + 300,
    });

    await gateway.handleConnection(client as never);

    expect(sessionAuthority.withActiveSession).not.toHaveBeenCalled();
    expect(client.join).not.toHaveBeenCalled();
    expect(client.disconnect).toHaveBeenCalledTimes(1);
  });

  it('rejects a realtime connection whose persisted session is revoked', async () => {
    const { gateway, client, sessionAuthority } = build();
    sessionAuthority.withActiveSession.mockResolvedValueOnce({ authorized: false });

    await gateway.handleConnection(client as never);

    expect(client.emit).toHaveBeenCalledWith('session:revoked', { reason: 'session_revoked' });
    expect(client.join).not.toHaveBeenCalled();
    expect(client.disconnect).toHaveBeenCalledTimes(1);
  });

  it('actively expires the socket lease and removes event authority at token expiry', async () => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-08-24T20:30:00.000Z'));

    try {
      const { gateway, client, jwtService, chatSecurity, realtimeAdmission } = build();
      jwtService.verifyAsync.mockResolvedValueOnce({
        sub: 'user-1',
        sid: 'session-1',
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
      expect(chatSecurity.persistMessageInTransaction).not.toHaveBeenCalled();
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
        sid: 'session-1',
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
      expect(chatSecurity.withAuthorizedRealtimeRecipientsInTransaction).not.toHaveBeenCalled();
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
        sid: 'session-1',
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

  it('rejects a revoked persisted session after admission and before persistence', async () => {
    const { gateway, client, sessionAuthority, chatSecurity, realtimeAdmission } = build();
    await gateway.handleConnection(client as never);
    sessionAuthority.withActiveSession.mockClear();
    sessionAuthority.withActiveSession.mockResolvedValueOnce({ authorized: false });

    await expect(
      gateway.handleMessage(client as never, {
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: 'hello',
      })
    ).resolves.toEqual({ success: false, error: 'unauthorized' });

    expect(realtimeAdmission.consume).toHaveBeenCalledWith('user-1', 'message');
    expect(sessionAuthority.withActiveSession).toHaveBeenCalledWith(
      'session-1',
      'user-1',
      expect.any(Function)
    );
    expect(client.emit).toHaveBeenCalledWith('session:revoked', { reason: 'session_revoked' });
    expect(client.disconnect).toHaveBeenCalledTimes(1);
    expect(chatSecurity.persistMessageInTransaction).not.toHaveBeenCalled();
  });

  it('threads the authority transaction through persistence and excludes revoked recipient sessions', async () => {
    const {
      gateway,
      client,
      makeClient,
      authorityTx,
      jwtService,
      sessionAuthority,
      chatSecurity,
      to,
      except,
      emit,
    } = build();
    await gateway.handleConnection(client as never);

    const user3Client = makeClient('socket-3');
    jwtService.verifyAsync.mockResolvedValueOnce({
      sub: 'user-3',
      sid: 'session-3',
      exp: Math.floor(Date.now() / 1_000) + 300,
    });
    await gateway.handleConnection(user3Client as never);

    const createdAt = new Date('2026-08-24T04:00:00.000Z');
    chatSecurity.persistMessageInTransaction.mockResolvedValue({
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
      async (_conversationId: string, deliver: (userIds: string[]) => void | Promise<void>) => {
        await deliver(['user-1', 'user-3']);
        return { authorizedUserIds: ['user-1', 'user-3'] };
      }
    );
    sessionAuthority.withActiveSessions.mockImplementationOnce(
      async (
        _sessionIds: string[],
        deliver: (activeSessionIds: Set<string>) => void | Promise<void>
      ) => {
        const activeSessionIds = new Set(['session-1']);
        await deliver(activeSessionIds);
        return { activeSessionIds };
      }
    );

    await expect(
      gateway.handleMessage(client as never, {
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: '  hello  ',
      })
    ).resolves.toMatchObject({ success: true, duplicate: false });

    expect(chatSecurity.persistMessageInTransaction).toHaveBeenCalledWith(authorityTx, {
      userId: 'user-1',
      conversationId: 'conversation-1',
      clientMessageId: 'client-message-123',
      text: 'hello',
    });
    expect(chatSecurity.persistMessage).not.toHaveBeenCalled();
    expect(sessionAuthority.withActiveSessions).toHaveBeenCalledWith(
      expect.arrayContaining(['session-1', 'session-3']),
      expect.any(Function)
    );
    expect(to).toHaveBeenCalledWith(['user:user-1', 'user:user-3']);
    expect(except).toHaveBeenCalledWith(['socket-3']);
    expect(emit).toHaveBeenCalledWith('message:received', {
      id: 'message-1',
      conversationId: 'conversation-1',
      senderId: 'user-1',
      text: 'hello',
      mediaUrls: [],
      timestamp: createdAt,
    });
  });

  it('rejects malformed messages before admission or persisted session work begins', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, sessionAuthority } = build();
    await gateway.handleConnection(client as never);
    sessionAuthority.withActiveSession.mockClear();

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
    expect(sessionAuthority.withActiveSession).not.toHaveBeenCalled();
    expect(chatSecurity.persistMessageInTransaction).not.toHaveBeenCalled();
  });

  it('rejects message floods before persisted session and persistence work begins', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, sessionAuthority } = build();
    await gateway.handleConnection(client as never);
    sessionAuthority.withActiveSession.mockClear();
    realtimeAdmission.consume.mockReturnValueOnce({ allowed: false, retryAfterMs: 750 });

    await expect(
      gateway.handleMessage(client as never, {
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: 'hello',
      })
    ).resolves.toEqual({ success: false, error: 'rate_limited', retryAfterMs: 750 });

    expect(sessionAuthority.withActiveSession).not.toHaveBeenCalled();
    expect(chatSecurity.persistMessageInTransaction).not.toHaveBeenCalled();
  });

  it('keeps typing relationship and recipient-session checks on the authority transaction', async () => {
    const {
      gateway,
      client,
      makeClient,
      authorityTx,
      jwtService,
      sessionAuthority,
      chatSecurity,
      realtimeAdmission,
      to,
      except,
      emit,
    } = build();
    await gateway.handleConnection(client as never);

    const user2Client = makeClient('socket-2');
    jwtService.verifyAsync.mockResolvedValueOnce({
      sub: 'user-2',
      sid: 'session-2',
      exp: Math.floor(Date.now() / 1_000) + 300,
    });
    await gateway.handleConnection(user2Client as never);

    chatSecurity.withAuthorizedRealtimeRecipientsInTransaction.mockImplementation(
      async (
        tx: unknown,
        _conversationId: string,
        deliver: (userIds: string[]) => void | Promise<void>,
        requiredUserId?: string
      ) => {
        expect(tx).toBe(authorityTx);
        expect(requiredUserId).toBe('user-1');
        await deliver(['user-1', 'user-2']);
        return { authorizedUserIds: ['user-1', 'user-2'] };
      }
    );

    await expect(
      gateway.handleTypingStart(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: true });

    expect(realtimeAdmission.consume).toHaveBeenCalledWith('user-1', 'typing');
    expect(sessionAuthority.withActiveSessionsInTransaction).toHaveBeenCalledWith(
      authorityTx,
      expect.arrayContaining(['session-1', 'session-2']),
      expect.any(Function)
    );
    expect(chatSecurity.withAuthorizedRealtimeRecipients).not.toHaveBeenCalled();
    expect(to).toHaveBeenCalledWith(['user:user-1', 'user:user-2']);
    expect(except).toHaveBeenCalledWith('socket-1');
    expect(emit).toHaveBeenCalledWith('typing:start', { userId: 'user-1' });
  });

  it('rejects malformed typing payloads before admission or persisted session work', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, sessionAuthority, to } = build();
    await gateway.handleConnection(client as never);
    sessionAuthority.withActiveSession.mockClear();

    await expect(gateway.handleTypingStart(client as never, null)).resolves.toEqual({
      success: false,
      error: 'invalid_payload',
    });

    expect(realtimeAdmission.consume).not.toHaveBeenCalled();
    expect(sessionAuthority.withActiveSession).not.toHaveBeenCalled();
    expect(chatSecurity.withAuthorizedRealtimeRecipientsInTransaction).not.toHaveBeenCalled();
    expect(to).not.toHaveBeenCalled();
  });

  it('rejects typing floods before persisted session and relationship authorization work', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, sessionAuthority, to } = build();
    await gateway.handleConnection(client as never);
    sessionAuthority.withActiveSession.mockClear();
    realtimeAdmission.consume.mockReturnValueOnce({ allowed: false, retryAfterMs: 500 });

    await expect(
      gateway.handleTypingStart(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: false, error: 'rate_limited', retryAfterMs: 500 });

    expect(sessionAuthority.withActiveSession).not.toHaveBeenCalled();
    expect(chatSecurity.withAuthorizedRealtimeRecipientsInTransaction).not.toHaveBeenCalled();
    expect(to).not.toHaveBeenCalled();
  });

  it('rejects malformed membership payloads before admission or persisted session checks', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, sessionAuthority } = build();
    await gateway.handleConnection(client as never);
    client.join.mockClear();
    sessionAuthority.withActiveSession.mockClear();

    await expect(
      gateway.handleJoinConversation(client as never, { conversationId: 'bad id' })
    ).resolves.toEqual({ success: false, error: 'invalid_payload' });

    expect(realtimeAdmission.consume).not.toHaveBeenCalled();
    expect(sessionAuthority.withActiveSession).not.toHaveBeenCalled();
    expect(chatSecurity.assertConversationAccessInTransaction).not.toHaveBeenCalled();
    expect(client.join).not.toHaveBeenCalled();
  });

  it('rate-limits room membership churn before persisted session access checks', async () => {
    const { gateway, client, chatSecurity, realtimeAdmission, sessionAuthority } = build();
    await gateway.handleConnection(client as never);
    client.join.mockClear();
    sessionAuthority.withActiveSession.mockClear();
    realtimeAdmission.consume.mockReturnValueOnce({ allowed: false, retryAfterMs: 400 });

    await expect(
      gateway.handleJoinConversation(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: false, error: 'rate_limited', retryAfterMs: 400 });

    expect(sessionAuthority.withActiveSession).not.toHaveBeenCalled();
    expect(chatSecurity.assertConversationAccessInTransaction).not.toHaveBeenCalled();
    expect(client.join).not.toHaveBeenCalled();
  });

  it('does not emit typing after relationship authorization is revoked', async () => {
    const { gateway, client, chatSecurity, to } = build();
    await gateway.handleConnection(client as never);
    chatSecurity.withAuthorizedRealtimeRecipientsInTransaction.mockRejectedValue(
      new ForbiddenException('Conversation is unavailable')
    );

    await expect(
      gateway.handleTypingStop(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: false, error: 'conversation_unavailable' });

    expect(to).not.toHaveBeenCalled();
  });
});
