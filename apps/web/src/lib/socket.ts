import { io, Socket } from 'socket.io-client';
import type { ChatMessage } from './api/chat';
import { useAuthStore } from './stores/auth-store';

let socket: Socket | null = null;

export function getSocket(): Socket {
  if (!socket) {
    const token = useAuthStore.getState().token;
    const apiUrl =
      process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:4000';

    socket = io(apiUrl, {
      auth: { token },
      autoConnect: false,
    });
  }

  return socket;
}

export function connectSocket() {
  const activeSocket = getSocket();
  activeSocket.auth = { token: useAuthStore.getState().token };
  if (!activeSocket.connected) activeSocket.connect();
  return activeSocket;
}

export function disconnectSocket() {
  if (socket?.connected) socket.disconnect();
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
    getSocket().emit('conversation:join', { conversationId });
  },
  leaveConversation: (conversationId: string) => {
    getSocket().emit('conversation:leave', { conversationId });
  },
  sendMessage: (conversationId: string, text: string, clientMessageId: string) =>
    new Promise<ChatMessage>((resolve, reject) => {
      getSocket()
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
    }),
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
    getSocket().emit('typing:start', { conversationId });
  },
  stopTyping: (conversationId: string) => {
    getSocket().emit('typing:stop', { conversationId });
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
