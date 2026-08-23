'use client';

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { ChevronLeft, Loader2, MapPin, Send } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  chatApi,
  type ChatConversation,
  type ChatMessage,
  type ChatMessagePage,
} from '@/lib/api/chat';
import {
  getOrCreateChatSendAttempt,
  invalidateAttemptForEditedDraft,
  reconcileDraftAfterAcknowledgedSend,
  type ChatSendAttempt,
} from '@/lib/chat-delivery';
import { meetupProposalsApi } from '@/lib/api/meetup-proposals';
import { chatSocket, connectSocket } from '@/lib/socket';
import { cn } from '@/lib/utils';
import { MeetupProposalSheet } from './meetup-proposal-sheet';

interface ChatWindowProps {
  conversation: ChatConversation;
  currentUserId: string;
  onBack: () => void;
}

export function ChatWindow({ conversation, currentUserId, onBack }: ChatWindowProps) {
  const queryClient = useQueryClient();
  const [inputValue, setInputValue] = useState('');
  const [proposalOpen, setProposalOpen] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [proposalMessage, setProposalMessage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pendingSendRef = useRef<ChatSendAttempt | null>(null);

  const messages = useQuery({
    queryKey: ['chat', 'messages', conversation.id],
    queryFn: () => chatApi.getMessages(conversation.id),
    retry: false,
  });

  useEffect(() => {
    const activeSocket = connectSocket();
    chatSocket.joinConversation(conversation.id);
    void chatApi.markRead(conversation.id).catch(() => undefined);

    const removeMessageListener = chatSocket.onMessage((message) => {
      if (message.conversationId !== conversation.id) return;
      queryClient.setQueryData<ChatMessagePage>(
        ['chat', 'messages', conversation.id],
        (current) => {
          if (!current) {
            return { data: [message], total: 1, page: 1, limit: 50 };
          }
          if (current.data.some((entry) => entry.id === message.id)) return current;
          return {
            ...current,
            data: [...current.data, message],
            total: current.total + 1,
          };
        }
      );
      void chatApi.markRead(conversation.id).catch(() => undefined);
      void queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
    });

    return () => {
      pendingSendRef.current = null;
      removeMessageListener();
      chatSocket.leaveConversation(conversation.id);
      if (activeSocket.connected) {
        chatSocket.stopTyping(conversation.id);
      }
    };
  }, [conversation.id, queryClient]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.data?.data]);

  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text || sending) return;

    const attempt = getOrCreateChatSendAttempt(pendingSendRef.current, conversation.id, text);
    pendingSendRef.current = attempt;
    setSending(true);
    setSendError(null);

    try {
      const persisted = await chatSocket.sendMessage(
        conversation.id,
        attempt.text,
        attempt.clientMessageId
      );
      if (pendingSendRef.current?.clientMessageId === attempt.clientMessageId) {
        pendingSendRef.current = null;
      }
      queryClient.setQueryData<ChatMessagePage>(
        ['chat', 'messages', conversation.id],
        (current) => {
          if (!current || current.data.some((entry) => entry.id === persisted.id)) return current;
          return { ...current, data: [...current.data, persisted], total: current.total + 1 };
        }
      );
      setInputValue((current) => reconcileDraftAfterAcknowledgedSend(current, attempt.text));
      void queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
    } catch {
      setSendError(
        'Delivery was not confirmed. Your draft is still here, and retrying it will not duplicate it.'
      );
    } finally {
      setSending(false);
    }
  };

  const handleProposalSubmit = async (data: {
    location: string;
    datetime: string;
    notes: string;
  }) => {
    setProposalMessage(null);
    try {
      const suggestedTime = new Date(data.datetime);
      await meetupProposalsApi.create({
        recipientId: conversation.participant.id,
        suggestedTime: suggestedTime.toISOString(),
        suggestedVenue: {
          name: data.location.trim(),
          type: 'public_place',
        },
        ...(data.notes.trim() ? { notes: data.notes.trim() } : {}),
      });
      setProposalOpen(false);
      setProposalMessage('Meetup suggestion sent. You can track it in Meetups.');
      await queryClient.invalidateQueries({ queryKey: ['meetup-proposals'] });
    } catch {
      setProposalMessage('That meetup suggestion was not saved. Check the time and try again.');
    }
  };

  const messageList: ChatMessage[] = messages.data?.data ?? [];

  return (
    <div className="flex h-[calc(100vh-5rem)] flex-col">
      <div className="glass-strong sticky top-[73px] z-30 border-b border-border/50">
        <div className="flex items-center gap-3 px-4 py-3">
          <Button variant="ghost" size="icon" onClick={onBack} aria-label="Back to conversations">
            <ChevronLeft className="h-5 w-5" aria-hidden="true" />
          </Button>
          <Avatar className="h-10 w-10">
            <AvatarImage src={conversation.participant.avatarUrl || '/placeholder.svg'} alt="" />
            <AvatarFallback>
              {conversation.participant.name.slice(0, 1).toUpperCase()}
            </AvatarFallback>
          </Avatar>
          <div className="min-w-0 flex-1">
            <h2 className="truncate font-semibold">
              {conversation.participant.name}
              {conversation.participant.petName ? ` & ${conversation.participant.petName}` : ''}
            </h2>
            <p className="text-xs text-muted-foreground">Private direct conversation</p>
          </div>
          <Button asChild size="sm" variant="outline" className="bg-transparent">
            <Link href="/meetups">Meetups</Link>
          </Button>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.isLoading ? (
          <div className="flex min-h-52 items-center justify-center gap-2" role="status">
            <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
            <span className="text-sm text-muted-foreground">Loading conversation…</span>
          </div>
        ) : messages.isError ? (
          <div className="surface-soft rounded-2xl p-5 text-center">
            <p className="font-semibold">Conversation history is unavailable</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Woof will not substitute a demo transcript while canonical messages cannot be read.
            </p>
            <Button
              variant="outline"
              className="mt-4 bg-transparent"
              onClick={() => messages.refetch()}
            >
              Try again
            </Button>
          </div>
        ) : messageList.length === 0 ? (
          <div className="surface-soft rounded-2xl p-5 text-center">
            <p className="font-semibold">Start with a simple hello</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Messages are private to members of this conversation. Once you have chatted, you can
              suggest a public-place meetup.
            </p>
          </div>
        ) : (
          messageList.map((message, index) => {
            const isCurrentUser = message.senderId === currentUserId;
            const previous = index > 0 ? messageList[index - 1] : null;
            const showTimestamp =
              !previous ||
              new Date(message.createdAt).getTime() - new Date(previous.createdAt).getTime() >
                60 * 60 * 1000;

            return (
              <div key={message.id} className="space-y-2">
                {showTimestamp && (
                  <div className="text-center">
                    <span className="text-xs text-muted-foreground">
                      {format(new Date(message.createdAt), 'MMM d, h:mm a')}
                    </span>
                  </div>
                )}
                <div className={cn('flex gap-2', isCurrentUser ? 'justify-end' : 'justify-start')}>
                  {!isCurrentUser && (
                    <Avatar className="h-8 w-8 shrink-0">
                      <AvatarImage
                        src={conversation.participant.avatarUrl || '/placeholder.svg'}
                        alt=""
                      />
                      <AvatarFallback>{conversation.participant.name.slice(0, 1)}</AvatarFallback>
                    </Avatar>
                  )}
                  <div
                    className={cn(
                      'max-w-[78%] rounded-2xl px-4 py-2',
                      isCurrentUser ? 'bg-primary text-primary-foreground' : 'glass'
                    )}
                  >
                    <p className="whitespace-pre-wrap break-words text-sm">{message.content}</p>
                  </div>
                </div>
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="glass-strong sticky bottom-16 border-t border-border/50 p-4">
        {proposalMessage && (
          <p className="mb-2 text-xs text-muted-foreground" role="status">
            {proposalMessage}
          </p>
        )}
        {sendError && (
          <p className="mb-2 text-xs text-destructive" role="alert">
            {sendError}
          </p>
        )}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setProposalOpen(true)}
            className="shrink-0 bg-transparent"
            aria-label="Suggest a meetup"
          >
            <MapPin className="h-5 w-5" aria-hidden="true" />
          </Button>
          <Input
            placeholder="Message…"
            value={inputValue}
            maxLength={4000}
            onChange={(event) => {
              const nextDraft = event.target.value;
              pendingSendRef.current = invalidateAttemptForEditedDraft(
                pendingSendRef.current,
                nextDraft
              );
              setInputValue(nextDraft);
              setSendError(null);
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void handleSend();
              }
            }}
            onFocus={() => chatSocket.startTyping(conversation.id)}
            onBlur={() => chatSocket.stopTyping(conversation.id)}
            className="flex-1"
          />
          <Button
            size="icon"
            onClick={() => void handleSend()}
            disabled={!inputValue.trim() || sending}
            aria-label="Send message"
          >
            {sending ? (
              <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
            ) : (
              <Send className="h-5 w-5" aria-hidden="true" />
            )}
          </Button>
        </div>
      </div>

      <MeetupProposalSheet
        open={proposalOpen}
        onOpenChange={setProposalOpen}
        onSubmit={(data) => void handleProposalSubmit(data)}
      />
    </div>
  );
}
