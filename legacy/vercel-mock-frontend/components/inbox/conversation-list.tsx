"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { formatDistanceToNow } from "date-fns"
import { cn } from "@/lib/utils"

interface Conversation {
  id: string
  matchId: string
  participant: {
    id: string
    name: string
    avatarUrl: string
    petName: string
  }
  lastMessage: {
    content: string
    timestamp: string
    senderId: string
    read: boolean
  }
  unreadCount: number
}

interface ConversationListProps {
  conversations: Conversation[]
  onSelectConversation: (id: string) => void
}

export function ConversationList({ conversations, onSelectConversation }: ConversationListProps) {
  if (conversations.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-4">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
          <span className="text-2xl">💬</span>
        </div>
        <h3 className="text-lg font-semibold mb-2">No conversations yet</h3>
        <p className="text-sm text-muted-foreground text-center">
          Start chatting with your matches to plan playdates and activities!
        </p>
      </div>
    )
  }

  return (
    <div className="divide-y divide-border">
      {conversations.map((conversation) => {
        const isUnread = conversation.unreadCount > 0
        const isFromOther = conversation.lastMessage.senderId !== "current-user"

        return (
          <button
            key={conversation.id}
            onClick={() => onSelectConversation(conversation.id)}
            className="w-full px-4 py-4 flex items-start gap-3 hover:bg-muted/50 transition-colors text-left"
          >
            <Avatar className="w-12 h-12 shrink-0">
              <AvatarImage src={conversation.participant.avatarUrl || "/placeholder.svg"} />
              <AvatarFallback>{conversation.participant.name[0]}</AvatarFallback>
            </Avatar>

            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-2 mb-1">
                <h3 className={cn("font-semibold truncate", isUnread && "text-primary")}>
                  {conversation.participant.name} & {conversation.participant.petName}
                </h3>
                <span className="text-xs text-muted-foreground shrink-0">
                  {formatDistanceToNow(new Date(conversation.lastMessage.timestamp), { addSuffix: true })}
                </span>
              </div>

              <div className="flex items-center justify-between gap-2">
                <p
                  className={cn(
                    "text-sm truncate",
                    isUnread && isFromOther ? "font-medium text-foreground" : "text-muted-foreground",
                  )}
                >
                  {isFromOther ? "" : "You: "}
                  {conversation.lastMessage.content}
                </p>
                {isUnread && (
                  <Badge variant="default" className="shrink-0 h-5 min-w-5 px-1.5 text-xs">
                    {conversation.unreadCount}
                  </Badge>
                )}
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
