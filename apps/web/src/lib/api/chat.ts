import { apiClient } from './client';

export type ChatConversation = {
  id: string;
  participant: {
    id: string;
    name: string;
    avatarUrl?: string | null;
    petId?: string | null;
    petName?: string | null;
    petAvatarUrl?: string | null;
  };
  lastMessage: {
    id: string;
    senderId: string;
    content: string;
    mediaUrls: string[];
    createdAt: string;
  } | null;
  unreadCount: number;
  updatedAt: string;
};

export type ChatMessage = {
  id: string;
  conversationId: string;
  senderId: string;
  content: string;
  type: 'text' | 'image';
  mediaUrl?: string | null;
  createdAt: string;
};

export type ChatMessagePage = {
  data: ChatMessage[];
  total: number;
  page: number;
  limit: number;
};

export const chatApi = {
  getConversations: () => apiClient.get<ChatConversation[]>('/chat/conversations'),
  createConversation: (participantId: string) =>
    apiClient.post<{ id: string; created: boolean }>('/chat/conversations', { participantId }),
  getMessages: (conversationId: string, page = 1, limit = 50) =>
    apiClient.get<ChatMessagePage>(`/chat/conversations/${conversationId}/messages`, {
      params: { page, limit },
    }),
  markRead: (conversationId: string) =>
    apiClient.post<{ ok: boolean }>(`/chat/conversations/${conversationId}/read`, {}),
};
