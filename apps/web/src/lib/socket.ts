import { io, Socket } from 'socket.io-client';
import { useAuthStore } from './stores/auth-store';

let socket: Socket | null = null;

export function getSocket(): Socket {
  if (!socket) {
    const token = useAuthStore.getState().token;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL?.replace('/api/v1', '') || 'http://localhost:4000';

    socket = io(apiUrl, {
      auth: {
        token,
      },
      autoConnect: false,
    });
  }

  return socket;
}

export function connectSocket() {
  const socket = getSocket();
  if (!socket.connected) {
    socket.connect();
  }
  return socket;
}

export function disconnectSocket() {
  if (socket?.connected) {
    socket.disconnect();
  }
}

// Chat event handlers
export const chatSocket = {
  joinConversation: (conversationId: string) => {
    getSocket().emit('conversation:join', { conversationId });
  },

  leaveConversation: (conversationId: string) => {
    getSocket().emit('conversation:leave', { conversationId });
  },

  sendMessage: (conversationId: string, text: string) => {
    getSocket().emit('message:send', { conversationId, text });
  },

  onMessage: (callback: (message: any) => void) => {
    getSocket().on('message:received', callback);
  },

  startTyping: (conversationId: string) => {
    getSocket().emit('typing:start', { conversationId });
  },

  stopTyping: (conversationId: string) => {
    getSocket().emit('typing:stop', { conversationId });
  },

  onTyping: (callback: (data: { userId: string }) => void) => {
    getSocket().on('typing:start', callback);
  },

  onStopTyping: (callback: (data: { userId: string }) => void) => {
    getSocket().on('typing:stop', callback);
  },
};
