export type ChatSendAttempt = {
  conversationId: string;
  text: string;
  clientMessageId: string;
};

export function createChatClientMessageId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 14)}`;
}

export function getOrCreateChatSendAttempt(
  existing: ChatSendAttempt | null,
  conversationId: string,
  draft: string,
  createId: () => string = createChatClientMessageId
): ChatSendAttempt {
  const text = draft.trim();
  if (existing && existing.conversationId === conversationId && existing.text === text) {
    return existing;
  }
  return {
    conversationId,
    text,
    clientMessageId: createId(),
  };
}

export function invalidateAttemptForEditedDraft(
  existing: ChatSendAttempt | null,
  nextDraft: string
): ChatSendAttempt | null {
  if (!existing) return null;
  return nextDraft.trim() === existing.text ? existing : null;
}

export function reconcileDraftAfterAcknowledgedSend(currentDraft: string, sentText: string) {
  return currentDraft.trim() === sentText ? '' : currentDraft;
}
