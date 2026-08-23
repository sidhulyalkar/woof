import { describe, expect, it, vi } from 'vitest';
import {
  getOrCreateChatSendAttempt,
  invalidateAttemptForEditedDraft,
  reconcileDraftAfterAcknowledgedSend,
} from './chat-delivery';

describe('chat delivery attempt identity', () => {
  it('reuses the same clientMessageId for an uncertain retry of the same draft', () => {
    const createId = vi.fn().mockReturnValueOnce('client-message-1');
    const first = getOrCreateChatSendAttempt(null, 'conversation-1', ' hello ', createId);
    const retry = getOrCreateChatSendAttempt(first, 'conversation-1', 'hello', createId);

    expect(first).toEqual({
      conversationId: 'conversation-1',
      text: 'hello',
      clientMessageId: 'client-message-1',
    });
    expect(retry).toBe(first);
    expect(createId).toHaveBeenCalledTimes(1);
  });

  it('creates a new identity when the draft changes or the conversation changes', () => {
    const createId = vi
      .fn()
      .mockReturnValueOnce('client-message-1')
      .mockReturnValueOnce('client-message-2')
      .mockReturnValueOnce('client-message-3');
    const first = getOrCreateChatSendAttempt(null, 'conversation-1', 'hello', createId);
    const edited = getOrCreateChatSendAttempt(first, 'conversation-1', 'hello again', createId);
    const moved = getOrCreateChatSendAttempt(edited, 'conversation-2', 'hello again', createId);

    expect(edited.clientMessageId).toBe('client-message-2');
    expect(moved.clientMessageId).toBe('client-message-3');
    expect(createId).toHaveBeenCalledTimes(3);
  });

  it('invalidates retry identity after an intentional draft edit', () => {
    const attempt = {
      conversationId: 'conversation-1',
      text: 'hello',
      clientMessageId: 'client-message-1',
    };

    expect(invalidateAttemptForEditedDraft(attempt, ' hello ')).toBe(attempt);
    expect(invalidateAttemptForEditedDraft(attempt, 'hello there')).toBeNull();
  });

  it('does not erase text typed after the acknowledged send started', () => {
    expect(reconcileDraftAfterAcknowledgedSend('hello', 'hello')).toBe('');
    expect(reconcileDraftAfterAcknowledgedSend('hello again', 'hello')).toBe('hello again');
  });
});
