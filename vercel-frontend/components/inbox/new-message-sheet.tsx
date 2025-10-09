"use client"

import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Search } from "lucide-react"

interface NewMessageSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSelectUser: (userId: string) => void
}

// Mock friends data for demonstration
const mockFriends = [
  {
    id: "f1",
    name: "Sarah Johnson",
    avatarUrl: "/woman-and-loyal-companion.png",
    petName: "Max",
    petAvatar: "/golden-retriever.png",
    location: "2.3 miles away",
    mutualFriends: 5,
    status: "friends" as const,
    friendsSince: new Date().toISOString(),
  },
  {
    id: "f2",
    name: "Michael Chen",
    avatarUrl: "/man-with-husky.jpg",
    petName: "Luna",
    petAvatar: "/siberian-husky-portrait.png",
    location: "1.8 miles away",
    mutualFriends: 3,
    status: "friends" as const,
    friendsSince: new Date().toISOString(),
  },
  {
    id: "f3",
    name: "Emily Rodriguez",
    avatarUrl: "/yoga-instructor.png",
    petName: "Bella",
    petAvatar: "/australian-shepherd-portrait.png",
    location: "3.1 miles away",
    mutualFriends: 7,
    status: "friends" as const,
    friendsSince: new Date().toISOString(),
  },
]

export function NewMessageSheet({ open, onOpenChange, onSelectUser }: NewMessageSheetProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const friends = mockFriends

  const filteredFriends = friends.filter(
    (friend) =>
      friend.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      friend.petName.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="bottom" className="h-[80vh]">
        <SheetHeader>
          <SheetTitle>New Message</SheetTitle>
        </SheetHeader>

        <div className="mt-4 space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search friends..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          <div className="space-y-1 overflow-y-auto max-h-[calc(80vh-140px)]">
            {filteredFriends.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>No friends found</p>
              </div>
            ) : (
              filteredFriends.map((friend) => (
                <button
                  key={friend.id}
                  onClick={() => onSelectUser(friend.id)}
                  className="w-full px-4 py-3 flex items-center gap-3 hover:bg-muted/50 transition-colors rounded-lg"
                >
                  <Avatar className="w-12 h-12">
                    <AvatarImage src={friend.avatarUrl || "/placeholder.svg"} />
                    <AvatarFallback>{friend.name[0]}</AvatarFallback>
                  </Avatar>
                  <div className="flex-1 text-left">
                    <h3 className="font-semibold">
                      {friend.name} & {friend.petName}
                    </h3>
                    <p className="text-sm text-muted-foreground">{friend.location}</p>
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
