export const MAX_CHAT_MESSAGE_LENGTH = 4_000;
export const MAX_CHAT_IDENTIFIER_LENGTH = 128;
export const MAX_REALTIME_PACKET_BYTES = 32 * 1024;

const CHAT_IDENTIFIER_PATTERN = /^[A-Za-z0-9_-]{8,128}$/;
const CLIENT_MESSAGE_ID_PATTERN = /^[A-Za-z0-9_-]{8,128}$/;

export type SendChatMessage = {
  conversationId: string;
  clientMessageId: string;
  text: string;
};

export type ConversationPayload = {
  conversationId: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isChatIdentifier(value: unknown): value is string {
  return typeof value === 'string' && CHAT_IDENTIFIER_PATTERN.test(value);
}

export function isClientMessageId(value: unknown): value is string {
  return typeof value === 'string' && CLIENT_MESSAGE_ID_PATTERN.test(value);
}

export function normalizeChatMessageText(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  if (value.length > MAX_CHAT_MESSAGE_LENGTH || value.includes('\u0000')) return null;

  const text = value.trim();
  return text.length > 0 ? text : null;
}

export function parseSendChatMessagePayload(value: unknown): SendChatMessage | null {
  if (!isRecord(value)) return null;
  if (!isChatIdentifier(value.conversationId)) return null;
  if (!isClientMessageId(value.clientMessageId)) return null;

  const text = normalizeChatMessageText(value.text);
  if (text === null) return null;

  return {
    conversationId: value.conversationId,
    clientMessageId: value.clientMessageId,
    text,
  };
}

export function parseConversationPayload(value: unknown): ConversationPayload | null {
  if (!isRecord(value) || !isChatIdentifier(value.conversationId)) return null;
  return { conversationId: value.conversationId };
}
