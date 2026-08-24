import {
  MAX_CHAT_MESSAGE_LENGTH,
  MAX_REALTIME_PACKET_BYTES,
  normalizeChatMessageText,
  parseConversationPayload,
  parseSendChatMessagePayload,
} from './chat-input-contract';

describe('chat input contract', () => {
  it('normalizes a valid message payload', () => {
    expect(
      parseSendChatMessagePayload({
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: '  hello  ',
      })
    ).toEqual({
      conversationId: 'conversation-1',
      clientMessageId: 'client-message-123',
      text: 'hello',
    });
  });

  it.each([null, [], 'message', 42])('rejects non-object message payloads: %p', (payload) => {
    expect(parseSendChatMessagePayload(payload)).toBeNull();
  });

  it('rejects oversized raw text before normalization', () => {
    expect(
      parseSendChatMessagePayload({
        conversationId: 'conversation-1',
        clientMessageId: 'client-message-123',
        text: 'x'.repeat(MAX_CHAT_MESSAGE_LENGTH + 1),
      })
    ).toBeNull();
  });

  it('rejects empty and PostgreSQL-incompatible NUL text', () => {
    expect(normalizeChatMessageText('   ')).toBeNull();
    expect(normalizeChatMessageText('hello\u0000world')).toBeNull();
  });

  it('rejects malformed conversation and retry identifiers', () => {
    expect(
      parseSendChatMessagePayload({
        conversationId: 'bad id',
        clientMessageId: 'client-message-123',
        text: 'hello',
      })
    ).toBeNull();
    expect(
      parseSendChatMessagePayload({
        conversationId: 'conversation-1',
        clientMessageId: 'short',
        text: 'hello',
      })
    ).toBeNull();
  });

  it('accepts only bounded conversation payloads', () => {
    expect(parseConversationPayload({ conversationId: 'conversation-1' })).toEqual({
      conversationId: 'conversation-1',
    });
    expect(parseConversationPayload(null)).toBeNull();
    expect(parseConversationPayload({ conversationId: 'x'.repeat(129) })).toBeNull();
  });

  it('keeps the transport ceiling comfortably above the application message contract', () => {
    expect(MAX_REALTIME_PACKET_BYTES).toBe(32 * 1024);
    expect(MAX_REALTIME_PACKET_BYTES).toBeGreaterThan(MAX_CHAT_MESSAGE_LENGTH * 4);
  });
});
