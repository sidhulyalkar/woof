import { beforeEach, describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({
  io: vi.fn(),
  logout: vi.fn(),
  token: 'token-1',
}));

vi.mock('socket.io-client', () => ({ io: harness.io }));
vi.mock('./stores/auth-store', () => ({
  useAuthStore: {
    getState: () => ({ token: harness.token, logout: harness.logout }),
  },
}));

type Listener = (...args: unknown[]) => void;

function createSocketHarness() {
  const persistent = new Map<string, Set<Listener>>();
  const once = new Map<string, Set<Listener>>();
  const outgoing = vi.fn();
  let connectionNumber = 0;

  const add = (collection: Map<string, Set<Listener>>, event: string, listener: Listener) => {
    const listeners = collection.get(event) ?? new Set<Listener>();
    listeners.add(listener);
    collection.set(event, listeners);
  };
  const remove = (collection: Map<string, Set<Listener>>, event: string, listener: Listener) => {
    collection.get(event)?.delete(listener);
  };
  const serverEmit = (event: string, ...args: unknown[]) => {
    const eventArgs =
      event === 'session:ready' && args.length === 0 ? [{ socketId: socket.id }] : args;
    for (const listener of persistent.get(event) ?? []) listener(...eventArgs);
    const oneTimeListeners = [...(once.get(event) ?? [])];
    once.delete(event);
    for (const listener of oneTimeListeners) listener(...eventArgs);
  };

  const socket = {
    id: 'socket-0',
    connected: false,
    auth: {} as Record<string, unknown>,
    on: vi.fn((event: string, listener: Listener) => {
      add(persistent, event, listener);
      return socket;
    }),
    once: vi.fn((event: string, listener: Listener) => {
      add(once, event, listener);
      return socket;
    }),
    off: vi.fn((event: string, listener: Listener) => {
      remove(persistent, event, listener);
      remove(once, event, listener);
      return socket;
    }),
    connect: vi.fn(() => {
      connectionNumber += 1;
      socket.id = `socket-${connectionNumber}`;
      socket.connected = true;
      serverEmit('connect');
      return socket;
    }),
    disconnect: vi.fn(() => {
      socket.connected = false;
      serverEmit('disconnect', 'io client disconnect');
      return socket;
    }),
    emit: outgoing,
    timeout: vi.fn(() => ({ emit: outgoing })),
  };

  return { socket, outgoing, serverEmit };
}

function installSuccessfulMessageAck(
  transport: ReturnType<typeof createSocketHarness>,
  id = 'message-1'
) {
  transport.outgoing.mockImplementation(
    (event: string, _payload: unknown, ack?: (error: Error | null, response?: unknown) => void) => {
      if (event === 'message:send') {
        ack?.(null, {
          success: true,
          duplicate: false,
          message: {
            id,
            conversationId: 'conversation-1',
            senderId: 'user-1',
            text: 'hello',
            mediaUrls: [],
            timestamp: '2026-08-25T03:00:00.000Z',
          },
        });
      }
      return transport.socket;
    }
  );
}

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}

function expectNoJoin(transport: ReturnType<typeof createSocketHarness>) {
  expect(transport.outgoing).not.toHaveBeenCalledWith('conversation:join', expect.anything());
}

function expectNoTyping(transport: ReturnType<typeof createSocketHarness>) {
  expect(transport.outgoing).not.toHaveBeenCalledWith('typing:start', expect.anything());
  expect(transport.outgoing).not.toHaveBeenCalledWith('typing:stop', expect.anything());
}

describe('realtime session readiness', () => {
  beforeEach(() => {
    vi.resetModules();
    harness.io.mockReset();
    harness.logout.mockReset();
    harness.token = 'token-1';
  });

  it('releases desired membership exactly once after persisted session authority becomes ready', async () => {
    const transport = createSocketHarness();
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket } = await import('./socket');

    chatSocket.joinConversation('conversation-1');
    await flushPromises();

    expect(transport.socket.connect).toHaveBeenCalledTimes(1);
    expectNoJoin(transport);

    transport.serverEmit('session:ready');
    await flushPromises();

    expect(transport.outgoing).toHaveBeenCalledTimes(1);
    expect(transport.outgoing).toHaveBeenCalledWith('conversation:join', {
      conversationId: 'conversation-1',
    });
  });

  it('does not replay desired membership when session ready is emitted twice', async () => {
    const transport = createSocketHarness();
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket } = await import('./socket');

    chatSocket.joinConversation('conversation-1');
    transport.serverEmit('session:ready');
    transport.serverEmit('session:ready');
    await flushPromises();

    expect(transport.outgoing).toHaveBeenCalledTimes(1);
    expect(transport.outgoing).toHaveBeenCalledWith('conversation:join', {
      conversationId: 'conversation-1',
    });
  });

  it('ignores stale readiness from a previous transport epoch', async () => {
    const transport = createSocketHarness();
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket, connectSocket } = await import('./socket');

    chatSocket.joinConversation('conversation-1');
    const firstSocketId = transport.socket.id;
    transport.serverEmit('session:ready');
    transport.outgoing.mockClear();

    transport.socket.disconnect();
    connectSocket();
    expect(transport.socket.id).not.toBe(firstSocketId);

    transport.serverEmit('session:ready', { socketId: firstSocketId });
    await flushPromises();
    expectNoJoin(transport);
    chatSocket.startTyping('conversation-1');
    expectNoTyping(transport);

    transport.serverEmit('session:ready');
    await flushPromises();
    expect(transport.outgoing).toHaveBeenCalledWith('conversation:join', {
      conversationId: 'conversation-1',
    });
  });

  it('cancels desired membership when leave occurs before authorization readiness', async () => {
    const transport = createSocketHarness();
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket } = await import('./socket');

    chatSocket.joinConversation('conversation-1');
    chatSocket.leaveConversation('conversation-1');
    transport.serverEmit('session:ready');
    await flushPromises();

    expectNoJoin(transport);
    expect(transport.outgoing).not.toHaveBeenCalledWith('conversation:leave', expect.anything());
  });

  it('does not send a message until the realtime session is authorized', async () => {
    const transport = createSocketHarness();
    installSuccessfulMessageAck(transport);
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket } = await import('./socket');

    const pending = chatSocket.sendMessage('conversation-1', 'hello', 'client-message-readiness-1');
    await flushPromises();
    expect(transport.outgoing).not.toHaveBeenCalledWith(
      'message:send',
      expect.anything(),
      expect.anything()
    );

    transport.serverEmit('session:ready');
    await expect(pending).resolves.toMatchObject({ id: 'message-1', content: 'hello' });
  });

  it('rejoins desired conversations once after each authorized reconnect', async () => {
    const transport = createSocketHarness();
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket, connectSocket } = await import('./socket');

    chatSocket.joinConversation('conversation-1');
    transport.serverEmit('session:ready');
    await flushPromises();
    expect(transport.outgoing).toHaveBeenCalledWith('conversation:join', {
      conversationId: 'conversation-1',
    });

    transport.outgoing.mockClear();
    transport.socket.disconnect();
    connectSocket();
    await flushPromises();
    expectNoJoin(transport);

    transport.serverEmit('session:ready');
    await flushPromises();
    expect(transport.outgoing).toHaveBeenCalledTimes(1);
    expect(transport.outgoing).toHaveBeenCalledWith('conversation:join', {
      conversationId: 'conversation-1',
    });
  });

  it('drops ephemeral typing while not ready instead of replaying stale presence', async () => {
    const transport = createSocketHarness();
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket } = await import('./socket');

    chatSocket.startTyping('conversation-1');
    chatSocket.stopTyping('conversation-1');
    await flushPromises();
    expectNoTyping(transport);

    transport.serverEmit('session:ready');
    await flushPromises();
    expectNoTyping(transport);

    chatSocket.startTyping('conversation-1');
    expect(transport.outgoing).toHaveBeenCalledWith('typing:start', {
      conversationId: 'conversation-1',
    });
  });

  it('forces a fresh identity boundary and drops desired rooms when the token changes', async () => {
    const transport = createSocketHarness();
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket, connectSocket } = await import('./socket');

    chatSocket.joinConversation('conversation-1');
    transport.serverEmit('session:ready');
    transport.outgoing.mockClear();

    harness.token = 'token-2';
    connectSocket();

    expect(transport.socket.disconnect).toHaveBeenCalledTimes(1);
    expect(transport.socket.connect).toHaveBeenCalledTimes(2);
    expect(transport.socket.auth).toEqual({ token: 'token-2' });

    chatSocket.startTyping('conversation-1');
    expectNoTyping(transport);
    transport.serverEmit('session:ready');
    await flushPromises();
    expectNoJoin(transport);

    chatSocket.startTyping('conversation-1');
    expect(transport.outgoing).toHaveBeenCalledWith('typing:start', {
      conversationId: 'conversation-1',
    });
  });

  it('rejects all concurrent message waiters when authorization never becomes ready', async () => {
    vi.useFakeTimers();
    try {
      const transport = createSocketHarness();
      harness.io.mockReturnValue(transport.socket);
      const { chatSocket } = await import('./socket');

      const first = chatSocket.sendMessage('conversation-1', 'hello', 'client-message-timeout-1');
      const second = chatSocket.sendMessage('conversation-1', 'hello', 'client-message-timeout-2');
      const firstRejected = expect(first).rejects.toThrow(
        'Realtime session authorization timed out'
      );
      const secondRejected = expect(second).rejects.toThrow(
        'Realtime session authorization timed out'
      );

      await vi.advanceTimersByTimeAsync(8_000);
      await firstRejected;
      await secondRejected;
      expect(transport.outgoing).not.toHaveBeenCalledWith(
        'message:send',
        expect.anything(),
        expect.anything()
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it('clears desired membership and local auth when an admitted session is revoked', async () => {
    const transport = createSocketHarness();
    harness.io.mockReturnValue(transport.socket);
    const { chatSocket, connectSocket } = await import('./socket');

    chatSocket.joinConversation('conversation-1');
    transport.serverEmit('session:ready');
    transport.outgoing.mockClear();

    transport.serverEmit('session:revoked', { reason: 'session_revoked' });
    expect(harness.logout).toHaveBeenCalledTimes(1);

    transport.socket.disconnect();
    harness.token = 'token-2';
    connectSocket();
    transport.serverEmit('session:ready');
    await flushPromises();

    expectNoJoin(transport);
  });
});
