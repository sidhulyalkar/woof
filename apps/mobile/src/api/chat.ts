import apiClient from './client';
import { ChatConversation, ChatMessage, PaginatedResponse } from '../types';

export const chatApi = {
  async getConversations(): Promise<ChatConversation[]> {
    return apiClient.get('/chat/conversations');
  },

  async getConversationById(id: string): Promise<ChatConversation> {
    return apiClient.get(`/chat/conversations/${id}`);
  },

  async getMessages(conversationId: string, page: number = 1, limit: number = 50): Promise<PaginatedResponse<ChatMessage>> {
    return apiClient.get(`/chat/conversations/${conversationId}/messages`, {
      params: { page, limit },
    });
  },

  async sendMessage(conversationId: string, content: string, type: 'text' | 'image' | 'location' = 'text'): Promise<ChatMessage> {
    return apiClient.post(`/chat/conversations/${conversationId}/messages`, {
      content,
      type,
    });
  },

  async markAsRead(conversationId: string): Promise<void> {
    return apiClient.post(`/chat/conversations/${conversationId}/read`);
  },

  async createConversation(participantIds: string[]): Promise<ChatConversation> {
    return apiClient.post('/chat/conversations', { participantIds });
  },

  async uploadMessageMedia(messageId: string, mediaUri: string): Promise<{ url: string }> {
    const formData = new FormData();
    formData.append('file', {
      uri: mediaUri,
      type: 'image/jpeg',
      name: 'message-media.jpg',
    } as any);

    return apiClient.post(`/chat/messages/${messageId}/media`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};
