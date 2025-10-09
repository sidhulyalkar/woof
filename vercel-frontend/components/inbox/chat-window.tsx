"use client"

import { useState, useRef, useEffect } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ChevronLeft, Send, MapPin, MoreVertical } from "lucide-react"
import { format } from "date-fns"
import { cn } from "@/lib/utils"
import { MeetupProposalCard } from "./meetup-proposal-card"
import { MeetupProposalSheet } from "./meetup-proposal-sheet"
import type { Message } from "@/lib/types"

interface ChatWindowProps {
  conversation: {
    id: string
    participant: {
      id: string
      name: string
      avatarUrl: string
      petName: string
    }
  }
  onBack: () => void
}

// Mock messages
const mockMessages: Message[] = [
  {
    id: "m1",
    conversationId: "c1",
    senderId: "o1",
    content: "Hi! I saw we're a great match. Max would love to meet your pup!",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    read: true,
    type: "text",
  },
  {
    id: "m2",
    conversationId: "c1",
    senderId: "current-user",
    content: "That would be amazing! We love meeting new friends. How about this weekend?",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 23).toISOString(),
    read: true,
    type: "text",
  },
  {
    id: "m3",
    conversationId: "c1",
    senderId: "o1",
    content: "Perfect! I know a great hiking trail nearby.",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 22).toISOString(),
    read: true,
    type: "text",
  },
  {
    id: "m4",
    conversationId: "c1",
    senderId: "o1",
    content: "",
    timestamp: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
    read: true,
    type: "meetup-proposal",
    metadata: {
      location: { lat: 37.7749, lng: -122.4194, name: "Golden Gate Park Trail" },
      datetime: new Date(Date.now() + 1000 * 60 * 60 * 24 * 2).toISOString(),
    },
  },
  {
    id: "m5",
    conversationId: "c1",
    senderId: "o1",
    content: "That sounds great! Max would love to join you for a hike this weekend.",
    timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
    read: false,
    type: "text",
  },
]

export function ChatWindow({ conversation, onBack }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>(mockMessages)
  const [inputValue, setInputValue] = useState("")
  const [proposalOpen, setProposalOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = () => {
    if (!inputValue.trim()) return

    const newMessage: Message = {
      id: `m${messages.length + 1}`,
      conversationId: conversation.id,
      senderId: "current-user",
      content: inputValue,
      timestamp: new Date().toISOString(),
      read: false,
      type: "text",
    }

    setMessages([...messages, newMessage])
    setInputValue("")
  }

  const handleProposalSubmit = (data: { location: string; datetime: string; notes: string }) => {
    const newMessage: Message = {
      id: `m${messages.length + 1}`,
      conversationId: conversation.id,
      senderId: "current-user",
      content: "",
      timestamp: new Date().toISOString(),
      read: false,
      type: "meetup-proposal",
      metadata: {
        location: { lat: 37.7749, lng: -122.4194, name: data.location },
        datetime: data.datetime,
      },
    }

    setMessages([...messages, newMessage])
    setProposalOpen(false)
  }

  return (
    <div className="flex flex-col h-[calc(100vh-5rem)]">
      {/* Chat Header */}
      <div className="sticky top-[73px] z-30 glass-strong border-b border-border/50">
        <div className="flex items-center gap-3 px-4 py-3">
          <Button variant="ghost" size="icon" onClick={onBack}>
            <ChevronLeft className="w-5 h-5" />
          </Button>
          <Avatar className="w-10 h-10">
            <AvatarImage src={conversation.participant.avatarUrl || "/placeholder.svg"} />
            <AvatarFallback>{conversation.participant.name[0]}</AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <h2 className="font-semibold truncate">
              {conversation.participant.name} & {conversation.participant.petName}
            </h2>
            <p className="text-xs text-muted-foreground">Active now</p>
          </div>
          <Button variant="ghost" size="icon">
            <MoreVertical className="w-5 h-5" />
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.map((message, index) => {
          const isCurrentUser = message.senderId === "current-user"
          const showTimestamp =
            index === 0 ||
            new Date(message.timestamp).getTime() - new Date(messages[index - 1].timestamp).getTime() > 1000 * 60 * 60

          return (
            <div key={message.id} className="space-y-2">
              {showTimestamp && (
                <div className="text-center">
                  <span className="text-xs text-muted-foreground">
                    {format(new Date(message.timestamp), "MMM d, h:mm a")}
                  </span>
                </div>
              )}

              {message.type === "meetup-proposal" ? (
                <div className={cn("flex", isCurrentUser ? "justify-end" : "justify-start")}>
                  <MeetupProposalCard
                    location={message.metadata?.location?.name || ""}
                    datetime={message.metadata?.datetime || ""}
                    isCurrentUser={isCurrentUser}
                  />
                </div>
              ) : (
                <div className={cn("flex gap-2", isCurrentUser ? "justify-end" : "justify-start")}>
                  {!isCurrentUser && (
                    <Avatar className="w-8 h-8 shrink-0">
                      <AvatarImage src={conversation.participant.avatarUrl || "/placeholder.svg"} />
                      <AvatarFallback>{conversation.participant.name[0]}</AvatarFallback>
                    </Avatar>
                  )}
                  <div
                    className={cn(
                      "max-w-[75%] rounded-2xl px-4 py-2",
                      isCurrentUser ? "bg-primary text-primary-foreground" : "glass",
                    )}
                  >
                    <p className="text-sm">{message.content}</p>
                  </div>
                </div>
              )}
            </div>
          )
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="sticky bottom-16 glass-strong border-t border-border/50 p-4">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setProposalOpen(true)}
            className="shrink-0 bg-transparent"
          >
            <MapPin className="w-5 h-5" />
          </Button>
          <Input
            placeholder="Type a message..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            className="flex-1"
          />
          <Button size="icon" onClick={handleSend} disabled={!inputValue.trim()}>
            <Send className="w-5 h-5" />
          </Button>
        </div>
      </div>

      <MeetupProposalSheet open={proposalOpen} onOpenChange={setProposalOpen} onSubmit={handleProposalSubmit} />
    </div>
  )
}
