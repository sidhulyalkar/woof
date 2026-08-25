import { ChatGateway } from './chat.gateway';

describe('ChatGateway session readiness', () => {
  function build() {
    const jwtService = {
      verifyAsync: jest.fn().mockResolvedValue({
        sub: 'user-1',
        sid: 'session-1',
        exp: Math.floor(Date.now() / 1_000) + 300,
      }),
    };
    const sessionAuthority = {
      withActiveSession: jest.fn(),
      withActiveSessions: jest.fn(),
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
      checkChatActivityNudges: jest.fn(),
    };
    const gateway = new ChatGateway(
      jwtService as never,
      sessionAuthority as never,
      chatSecurity as never,
      realtimeAdmission as never,
      nudgesService as never
    );
    gateway.server = { to: jest.fn() } as never;

    const client: {
      id: string;
      connected: boolean;
      handshake: { auth: { token: string } };
      join: jest.Mock;
      leave: jest.Mock;
      emit: jest.Mock;
      disconnect: jest.Mock;
    } = {
      id: 'socket-1',
      connected: true,
      handshake: { auth: { token: 'token-1' } },
      join: jest.fn().mockResolvedValue(undefined),
      leave: jest.fn().mockResolvedValue(undefined),
      emit: jest.fn(),
      disconnect: jest.fn(),
    };
    client.disconnect.mockImplementation(() => {
      client.connected = false;
    });

    return {
      gateway,
      client,
      jwtService,
      sessionAuthority,
      chatSecurity,
      realtimeAdmission,
    };
  }

  it('emits session:ready only after persisted authority work and the private user-room join finish', async () => {
    const { gateway, client, sessionAuthority } = build();
    sessionAuthority.withActiveSession.mockImplementation(
      async (_sessionId: string, _userId: string, work: () => Promise<unknown>) => {
        const result = await work();
        expect(client.join).toHaveBeenCalledWith('user:user-1');
        expect(client.emit).not.toHaveBeenCalledWith('session:ready', expect.anything());
        return { authorized: true, result };
      }
    );

    await gateway.handleConnection(client as never);

    expect(sessionAuthority.withActiveSession).toHaveBeenCalledWith(
      'session-1',
      'user-1',
      expect.any(Function)
    );
    expect(client.emit).toHaveBeenCalledWith('session:ready', { socketId: 'socket-1' });
    expect(client.disconnect).not.toHaveBeenCalled();
  });

  it('never emits readiness for a revoked persisted session', async () => {
    const { gateway, client, sessionAuthority } = build();
    sessionAuthority.withActiveSession.mockResolvedValue({ authorized: false });

    await gateway.handleConnection(client as never);

    expect(client.emit).not.toHaveBeenCalledWith('session:ready', expect.anything());
    expect(client.emit).toHaveBeenCalledWith('session:revoked', { reason: 'session_revoked' });
    expect(client.disconnect).toHaveBeenCalledTimes(1);
  });

  it('does not create realtime authority when the transport disconnects before admission work starts', async () => {
    const { gateway, client, sessionAuthority, realtimeAdmission, chatSecurity } = build();
    sessionAuthority.withActiveSession.mockImplementation(
      async (_sessionId: string, _userId: string, work: () => Promise<unknown>) => {
        client.connected = false;
        return { authorized: true, result: await work() };
      }
    );

    await gateway.handleConnection(client as never);

    expect(client.join).not.toHaveBeenCalled();
    expect(client.emit).not.toHaveBeenCalledWith('session:ready', expect.anything());
    await expect(
      gateway.handleMessage(client as never, {
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-after-disconnect',
        text: 'hello',
      })
    ).resolves.toEqual({ success: false, error: 'unauthorized' });
    expect(realtimeAdmission.consume).not.toHaveBeenCalled();
    expect(chatSecurity.persistMessage).not.toHaveBeenCalled();
  });

  it('clears partial authority if the transport disconnects while the user room is joining', async () => {
    const { gateway, client, sessionAuthority, realtimeAdmission } = build();
    sessionAuthority.withActiveSession.mockImplementation(
      async (_sessionId: string, _userId: string, work: () => Promise<unknown>) => ({
        authorized: true,
        result: await work(),
      })
    );
    client.join.mockImplementationOnce(async () => {
      client.connected = false;
    });

    await gateway.handleConnection(client as never);

    expect(client.emit).not.toHaveBeenCalledWith('session:ready', expect.anything());
    await expect(
      gateway.handleTypingStart(client as never, { conversationId: 'conversation-1' })
    ).resolves.toEqual({ success: false, error: 'unauthorized' });
    expect(realtimeAdmission.consume).not.toHaveBeenCalled();
  });
});
