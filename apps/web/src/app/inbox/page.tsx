"use client"

import { useState } from "react"
import { useSearchParams } from "next/navigation"
import { BottomNav } from "@/components/bottom-nav"
import { ConversationList } from "@/components/inbox/conversation-list"
import { ChatWindow } from "@/components/inbox/chat-window"
import { FriendsListView } from "@/components/inbox/friends-list-view"
import { NewMessageSheet } from "@/components/inbox/new-message-sheet"
import { Search, Plus } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

// Mock conversations data
const mockConversations = [
  {
    id: "c1",
    matchId: "1",
    participant: {
      id: "o1",
      name: "Sarah Johnson",
      avatarUrl: "/woman-and-loyal-companion.png",
      petName: "Max",
    },
    lastMessage: {
      content: "That sounds great! Max would love to join you for a hike this weekend.",
      timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      senderId: "o1",
      read: false,
    },
    unreadCount: 2,
  },
  {
    id: "c2",
    matchId: "2",
    participant: {
      id: "o2",
      name: "Michael Chen",
      avatarUrl: "/man-with-husky.jpg",
      petName: "Luna",
    },
    lastMessage: {
      content: "Luna is super excited! See you at the dog park tomorrow at 10am.",
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
      senderId: "current-user",
      read: true,
    },
    unreadCount: 0,
  },
  {
    id: "c3",
    matchId: "3",
    participant: {
      id: "o3",
      name: "Emily Rodriguez",
      avatarUrl: "/yoga-instructor.png",
      petName: "Bella",
    },
    lastMessage: {
      content: "Thanks for the recommendation! I'll check out that pet cafe.",
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
      senderId: "o3",
      read: true,
    },
    unreadCount: 0,
  },
]

export default function InboxPage() {
  const searchParams = useSearchParams()
  const matchParam = searchParams.get("match")
  const [selectedConversation, setSelectedConversation] = useState<string | null>(matchParam || null)
  const [searchQuery, setSearchQuery] = useState("")
  const [showNewMessage, setShowNewMessage] = useState(false)
  const [activeTab, setActiveTab] = useState("messages")

  const filteredConversations = mockConversations.filter((conv) =>
    conv.participant.name.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const activeConversation = mockConversations.find((c) => c.id === selectedConversation)

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="px-4 py-4 max-w-lg mx-auto space-y-3">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">Messages</h1>
            <Button size="sm" onClick={() => setShowNewMessage(true)} className="gap-2">
              <Plus className="w-4 h-4" />
              New Message
            </Button>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="w-full">
              <TabsTrigger value="messages" className="flex-1">
                Messages
                {filteredConversations.filter((c) => c.unreadCount > 0).length > 0 && (
                  <span className="ml-2 px-1.5 py-0.5 text-xs bg-primary text-primary-foreground rounded-full">
                    {filteredConversations.filter((c) => c.unreadCount > 0).length}
                  </span>
                )}
              </TabsTrigger>
              <TabsTrigger value="friends" className="flex-1">
                Friends
              </TabsTrigger>
            </TabsList>
          </Tabs>

          {activeTab === "messages" && (
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search conversations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
          )}
        </div>
      </header>

      {/* Content */}
      <main className="max-w-lg mx-auto">
        {selectedConversation && activeConversation ? (
          <ChatWindow conversation={activeConversation} onBack={() => setSelectedConversation(null)} />
        ) : (
          <>
            {activeTab === "messages" && (
              <ConversationList conversations={filteredConversations} onSelectConversation={setSelectedConversation} />
            )}
            {activeTab === "friends" && <FriendsListView />}
          </>
        )}
      </main>

      <NewMessageSheet
        open={showNewMessage}
        onOpenChange={setShowNewMessage}
        onSelectUser={(userId) => {
          // Create a new conversation with the selected user
          console.log("[v0] Starting conversation with user:", userId)
          setShowNewMessage(false)
        }}
      />

      <BottomNav />
    </div>
  )
}
