'use client';

import { formatDistanceToNow } from 'date-fns';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import type { ChatConversation } from '@/lib/api/chat';
import { cn } from '@/lib/utils';

interface ConversationListProps {
  conversations: ChatConversation[];
  currentUserId: string;
  onSelectConversation: (id: string) => void;
}

export function ConversationList({
  conversations,
  currentUserId,
  onSelectConversation,
}: ConversationListProps) {
  if (conversations.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center px-4 py-16">
        <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
          <span className="text-2xl" aria-hidden="true">
            💬
          </span>
        </div>
        <h3 className="mb-2 text-lg font-semibold">No conversations yet</h3>
        <p className="max-w-sm text-center text-sm text-muted-foreground">
          Open an explainable match in Discover and start with a simple hello. Woof does not seed fake conversations.
        </p>
      </div>
    );
  }

  return (
    <div className="divide-y divide-border">
      {conversations.map((conversation) => {
        const isUnread = conversation.unreadCount > 0;
        const lastMessage = conversation.lastMessage;
        const isFromOther = lastMessage ? lastMessage.senderId !== currentUserId : false;
        const displayName = conversation.participant.petName
          ? `${conversation.participant.name} & ${conversation.participant.petName}`
          : conversation.participant.name;

        return (
          <button
            key={conversation.id}
            type="button"
            onClick={() => onSelectConversation(conversation.id)}
            className="flex w-full items-start gap-3 px-4 py-4 text-left transition-colors hover:bg-muted/50"
          >
            <Avatar className="h-12 w-12 shrink-0">
              <AvatarImage src={conversation.participant.avatarUrl || '/placeholder.svg'} alt="" />
              <AvatarFallback>{conversation.participant.name.slice(0, 1).toUpperCase()}</AvatarFallback>
            </Avatar>

            <div className="min-w-0 flex-1">
              <div className="mb-1 flex items-start justify-between gap-2">
                <h3 className={cn('truncate font-semibold', isUnread && 'text-primary')}>
                  {displayName}
                </h3>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {formatDistanceToNow(new Date(lastMessage?.createdAt ?? conversation.updatedAt), {
                    addSuffix: true,
                  })}
                </span>
              </div>

              <div className="flex items-center justify-between gap-2">
                <p
                  className={cn(
                    'truncate text-sm',
                    isUnread && isFromOther
                      ? 'font-medium text-foreground'
                      : 'text-muted-foreground',
                  )}
                >
                  {lastMessage ? (
                    <>
                      {!isFromOther ? 'You: ' : ''}
                      {lastMessage.content}
                    </>
                  ) : (
                    'No messages yet'
                  )}
                </p>
                {isUnread && (
                  <Badge variant="default" className="h-5 min-w-5 shrink-0 px-1.5 text-xs">
                    {conversation.unreadCount}
                  </Badge>
                )}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
