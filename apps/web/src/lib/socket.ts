import { io, Socket } from 'socket.io-client';
import type { ChatMessage } from './api/chat';
import { useAuthStore } from './stores/auth-store';

const SESSION_READY_TIMEOUT_MS = 8_000;

let socket: Socket | null = null;
let sessionReady = false;
let activeToken: string | null = null;
const desiredConversations = new Set<string>();

type PendingReadiness = {
  promise: Promise<Socket>;
  resolve: (socket: Socket) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

type SessionReadyPayload = {
  socketId?: unknown;
};

let pendingReadiness: PendingReadiness | null = null;

function settleReadiness(error?: Error) {
  const pending = pendingReadiness;
  if (!pending) return;

  pendingReadiness = null;
  clearTimeout(pending.timer);
  if (error) pending.reject(error);
  else if (socket) pending.resolve(socket);
}

function replayDesiredConversations(activeSocket: Socket) {
  for (const conversationId of desiredConversations) {
    activeSocket.emit('conversation:join', { conversationId });
  }
}

function clearRevokedAuth() {
  sessionReady = false;
  activeToken = null;
  desiredConversations.clear();
  settleReadiness(new Error('Realtime session is no longer authorized'));
  useAuthStore.getState().logout();
}

export function getSocket(): Socket {
  if (!socket) {
    const token = useAuthStore.getState().token;
    const apiUrl =
      process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:4000';

    activeToken = token;
    const activeSocket = io(apiUrl, {
      auth: { token },
      autoConnect: false,
    });
    socket = activeSocket;

    activeSocket.on('connect', () => {
      sessionReady = false;
    });
    activeSocket.on('session:ready', (payload: SessionReadyPayload) => {
      if (sessionReady || payload?.socketId !== activeSocket.id) return;
      sessionReady = true;
      replayDesiredConversations(activeSocket);
      settleReadiness();
    });
    activeSocket.on('disconnect', () => {
      sessionReady = false;
      settleReadiness(new Error('Realtime session disconnected before authorization'));
    });
    activeSocket.on('connect_error', (error: Error) => {
      sessionReady = false;
      settleReadiness(error);
    });
    activeSocket.on('session:expired', clearRevokedAuth);
    activeSocket.on('session:revoked', clearRevokedAuth);
  }

  return socket;
}

export function connectSocket() {
  const activeSocket = getSocket();
  const token = useAuthStore.getState().token;
  const tokenChanged = activeToken !== token;

  if (tokenChanged) {
    sessionReady = false;
    desiredConversations.clear();
    settleReadiness(new Error('Realtime credentials changed before authorization'));
    activeSocket.disconnect();
  }

  activeToken = token;
  activeSocket.auth = { token };
  if (!activeSocket.connected) {
    sessionReady = false;
    activeSocket.connect();
  }
  return activeSocket;
}

export function disconnectSocket() {
  sessionReady = false;
  activeToken = null;
  desiredConversations.clear();
  settleReadiness(new Error('Realtime session disconnected'));
  socket?.disconnect();
}

function waitForSessionReady(): Promise<Socket> {
  const activeSocket = connectSocket();
  if (activeSocket.connected && sessionReady) return Promise.resolve(activeSocket);
  if (pendingReadiness) return pendingReadiness.promise;

  let resolveReadiness!: (socket: Socket) => void;
  let rejectReadiness!: (error: Error) => void;
  const promise = new Promise<Socket>((resolve, reject) => {
    resolveReadiness = resolve;
    rejectReadiness = reject;
  });
  const timer = setTimeout(() => {
    if (pendingReadiness?.promise !== promise) return;
    settleReadiness(new Error('Realtime session authorization timed out'));
  }, SESSION_READY_TIMEOUT_MS);

  pendingReadiness = {
    promise,
    resolve: resolveReadiness,
    reject: rejectReadiness,
    timer,
  };

  if (activeSocket.connected && sessionReady) settleReadiness();
  return promise;
}

type SendAck = {
  success: boolean;
  duplicate?: boolean;
  error?: string;
  message?: {
    id: string;
    conversationId: string;
    senderId: string;
    text: string;
    mediaUrls: string[];
    timestamp: string;
  };
};

export const chatSocket = {
  joinConversation: (conversationId: string) => {
    desiredConversations.add(conversationId);
    const activeSocket = connectSocket();
    if (activeSocket.connected && sessionReady) {
      activeSocket.emit('conversation:join', { conversationId });
    }
  },
  leaveConversation: (conversationId: string) => {
    desiredConversations.delete(conversationId);
    const activeSocket = getSocket();
    if (activeSocket.connected && sessionReady) {
      activeSocket.emit('conversation:leave', { conversationId });
    }
  },
  sendMessage: (conversationId: string, text: string, clientMessageId: string) =>
    waitForSessionReady().then(
      (activeSocket) =>
        new Promise<ChatMessage>((resolve, reject) => {
          activeSocket
            .timeout(8_000)
            .emit(
              'message:send',
              { conversationId, clientMessageId, text },
              (timeoutError: Error | null, ack?: SendAck) => {
                if (timeoutError || !ack?.success || !ack.message) {
                  reject(timeoutError ?? new Error(ack?.error ?? 'Message was not accepted'));
                  return;
                }
                resolve({
                  id: ack.message.id,
                  conversationId: ack.message.conversationId,
                  senderId: ack.message.senderId,
                  content: ack.message.text,
                  type: ack.message.mediaUrls.length > 0 ? 'image' : 'text',
                  mediaUrl: ack.message.mediaUrls[0] ?? null,
                  createdAt: ack.message.timestamp,
                });
              }
            );
        })
    ),
  onMessage: (callback: (message: ChatMessage) => void) => {
    const handler = (message: {
      id: string;
      conversationId: string;
      senderId: string;
      text: string;
      mediaUrls: string[];
      timestamp: string;
    }) =>
      callback({
        id: message.id,
        conversationId: message.conversationId,
        senderId: message.senderId,
        content: message.text,
        type: message.mediaUrls.length > 0 ? 'image' : 'text',
        mediaUrl: message.mediaUrls[0] ?? null,
        createdAt: message.timestamp,
      });
    getSocket().on('message:received', handler);
    return () => getSocket().off('message:received', handler);
  },
  startTyping: (conversationId: string) => {
    const activeSocket = connectSocket();
    if (activeSocket.connected && sessionReady) {
      activeSocket.emit('typing:start', { conversationId });
    }
  },
  stopTyping: (conversationId: string) => {
    const activeSocket = getSocket();
    if (activeSocket.connected && sessionReady) {
      activeSocket.emit('typing:stop', { conversationId });
    }
  },
  onTyping: (callback: (data: { userId: string }) => void) => {
    getSocket().on('typing:start', callback);
    return () => getSocket().off('typing:start', callback);
  },
  onStopTyping: (callback: (data: { userId: string }) => void) => {
    getSocket().on('typing:stop', callback);
    return () => getSocket().off('typing:stop', callback);
  },
};
