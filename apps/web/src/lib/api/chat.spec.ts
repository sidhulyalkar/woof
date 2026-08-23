import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('./client', () => ({ apiClient: transport }));

import { chatApi } from './chat';

describe('chatApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
  });

  it('creates a direct conversation from a member id rather than a client-owned room id', async () => {
    transport.post.mockResolvedValue({ id: 'conversation-1', created: true });

    await chatApi.createConversation('member-2');

    expect(transport.post).toHaveBeenCalledWith('/chat/conversations', {
      participantId: 'member-2',
    });
  });

  it('reads bounded canonical message history', async () => {
    transport.get.mockResolvedValue({ data: [], total: 0, page: 1, limit: 50 });

    await chatApi.getMessages('conversation-1');

    expect(transport.get).toHaveBeenCalledWith('/chat/conversations/conversation-1/messages', {
      params: { page: 1, limit: 50 },
    });
  });

  it('advances the authenticated participant read watermark', async () => {
    transport.post.mockResolvedValue({ ok: true });

    await chatApi.markRead('conversation-1');

    expect(transport.post).toHaveBeenCalledWith('/chat/conversations/conversation-1/read', {});
  });
});
