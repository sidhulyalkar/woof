"use client"

import { useState } from "react"
import { Users, UserPlus, Search, Check, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BottomNav } from "@/components/bottom-nav"
import type { Friend, FriendRequest } from "@/lib/types"

export default function FriendsPage() {
  const [searchQuery, setSearchQuery] = useState("")

  // Mock friends data
  const friends: Friend[] = [
    {
      id: "1",
      name: "Sarah Chen",
      avatarUrl: "/user-avatar.jpg",
      petName: "Luna",
      petAvatar: "/border-collie.jpg",
      location: "San Francisco, CA",
      mutualFriends: 5,
      status: "friends",
      friendsSince: "2024-01-15",
    },
    {
      id: "2",
      name: "Mike Rodriguez",
      avatarUrl: "/man-with-husky.jpg",
      petName: "Max",
      petAvatar: "/siberian-husky-portrait.png",
      location: "Oakland, CA",
      mutualFriends: 3,
      status: "friends",
      friendsSince: "2024-02-01",
    },
    {
      id: "3",
      name: "Emma Wilson",
      avatarUrl: "/woman-and-loyal-companion.png",
      petName: "Charlie",
      petAvatar: "/golden-retriever.png",
      location: "Berkeley, CA",
      mutualFriends: 7,
      status: "friends",
      friendsSince: "2024-01-20",
    },
  ]

  const requests: FriendRequest[] = [
    {
      id: "r1",
      fromUserId: "4",
      fromUserName: "Alex Kim",
      fromUserAvatar: "/yoga-instructor.png",
      fromPetName: "Bella",
      fromPetAvatar: "/australian-shepherd-portrait.png",
      message: "Hey! Our dogs would love to play together!",
      timestamp: "2024-03-15T10:30:00Z",
    },
  ]

  const suggestions: Friend[] = [
    {
      id: "5",
      name: "Jessica Park",
      avatarUrl: "/placeholder.svg?height=100&width=100",
      petName: "Rocky",
      petAvatar: "/placeholder.svg?height=100&width=100",
      location: "San Francisco, CA",
      mutualFriends: 4,
      status: "none",
    },
    {
      id: "6",
      name: "David Lee",
      avatarUrl: "/placeholder.svg?height=100&width=100",
      petName: "Daisy",
      petAvatar: "/placeholder.svg?height=100&width=100",
      location: "Palo Alto, CA",
      mutualFriends: 2,
      status: "none",
    },
  ]

  const filteredFriends = friends.filter((friend) => friend.name.toLowerCase().includes(searchQuery.toLowerCase()))

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-border/40 bg-background/80 backdrop-blur-xl">
        <div className="flex items-center gap-3 px-4 py-4">
          <Users className="h-6 w-6 text-accent" />
          <h1 className="text-xl font-bold">Friends</h1>
        </div>

        {/* Search */}
        <div className="px-4 pb-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search friends..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>
      </div>

      <Tabs defaultValue="all" className="w-full">
        <TabsList className="w-full justify-start rounded-none border-b border-border/40 bg-transparent px-4">
          <TabsTrigger value="all">All Friends ({friends.length})</TabsTrigger>
          <TabsTrigger value="requests">Requests {requests.length > 0 && `(${requests.length})`}</TabsTrigger>
          <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
        </TabsList>

        {/* All Friends */}
        <TabsContent value="all" className="mt-0 space-y-2 p-4">
          {filteredFriends.length === 0 ? (
            <div className="py-12 text-center">
              <Users className="mx-auto h-12 w-12 text-muted-foreground/50" />
              <p className="mt-4 text-sm text-muted-foreground">
                {searchQuery ? "No friends found" : "No friends yet"}
              </p>
            </div>
          ) : (
            filteredFriends.map((friend) => (
              <div
                key={friend.id}
                className="flex items-center gap-4 rounded-xl border border-border/40 bg-card/50 p-4"
              >
                <div className="relative">
                  <Avatar className="h-12 w-12 border-2 border-border">
                    <AvatarImage src={friend.avatarUrl || "/placeholder.svg"} alt={friend.name} />
                    <AvatarFallback>{friend.name[0]}</AvatarFallback>
                  </Avatar>
                  <Avatar className="absolute -bottom-1 -right-1 h-6 w-6 border-2 border-background">
                    <AvatarImage src={friend.petAvatar || "/placeholder.svg"} alt={friend.petName} />
                    <AvatarFallback>{friend.petName[0]}</AvatarFallback>
                  </Avatar>
                </div>

                <div className="flex-1 min-w-0">
                  <p className="font-semibold truncate">{friend.name}</p>
                  <p className="text-sm text-muted-foreground truncate">with {friend.petName}</p>
                  <p className="text-xs text-muted-foreground">{friend.mutualFriends} mutual friends</p>
                </div>

                <Button variant="outline" size="sm">
                  View
                </Button>
              </div>
            ))
          )}
        </TabsContent>

        {/* Friend Requests */}
        <TabsContent value="requests" className="mt-0 space-y-3 p-4">
          {requests.length === 0 ? (
            <div className="py-12 text-center">
              <UserPlus className="mx-auto h-12 w-12 text-muted-foreground/50" />
              <p className="mt-4 text-sm text-muted-foreground">No pending requests</p>
            </div>
          ) : (
            requests.map((request) => (
              <div key={request.id} className="rounded-xl border border-border/40 bg-card/50 p-4 space-y-3">
                <div className="flex items-start gap-4">
                  <div className="relative">
                    <Avatar className="h-12 w-12 border-2 border-border">
                      <AvatarImage src={request.fromUserAvatar || "/placeholder.svg"} alt={request.fromUserName} />
                      <AvatarFallback>{request.fromUserName[0]}</AvatarFallback>
                    </Avatar>
                    <Avatar className="absolute -bottom-1 -right-1 h-6 w-6 border-2 border-background">
                      <AvatarImage src={request.fromPetAvatar || "/placeholder.svg"} alt={request.fromPetName} />
                      <AvatarFallback>{request.fromPetName[0]}</AvatarFallback>
                    </Avatar>
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="font-semibold">{request.fromUserName}</p>
                    <p className="text-sm text-muted-foreground">with {request.fromPetName}</p>
                    {request.message && <p className="mt-2 text-sm text-muted-foreground">"{request.message}"</p>}
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button className="flex-1 gap-2" size="sm">
                    <Check className="h-4 w-4" />
                    Accept
                  </Button>
                  <Button variant="outline" className="flex-1 gap-2 bg-transparent" size="sm">
                    <X className="h-4 w-4" />
                    Decline
                  </Button>
                </div>
              </div>
            ))
          )}
        </TabsContent>

        {/* Suggestions */}
        <TabsContent value="suggestions" className="mt-0 space-y-2 p-4">
          {suggestions.map((suggestion) => (
            <div
              key={suggestion.id}
              className="flex items-center gap-4 rounded-xl border border-border/40 bg-card/50 p-4"
            >
              <div className="relative">
                <Avatar className="h-12 w-12 border-2 border-border">
                  <AvatarImage src={suggestion.avatarUrl || "/placeholder.svg"} alt={suggestion.name} />
                  <AvatarFallback>{suggestion.name[0]}</AvatarFallback>
                </Avatar>
                <Avatar className="absolute -bottom-1 -right-1 h-6 w-6 border-2 border-background">
                  <AvatarImage src={suggestion.petAvatar || "/placeholder.svg"} alt={suggestion.petName} />
                  <AvatarFallback>{suggestion.petName[0]}</AvatarFallback>
                </Avatar>
              </div>

              <div className="flex-1 min-w-0">
                <p className="font-semibold truncate">{suggestion.name}</p>
                <p className="text-sm text-muted-foreground truncate">with {suggestion.petName}</p>
                <p className="text-xs text-muted-foreground">{suggestion.mutualFriends} mutual friends</p>
              </div>

              <Button variant="outline" size="sm" className="gap-2 bg-transparent">
                <UserPlus className="h-4 w-4" />
                Add
              </Button>
            </div>
          ))}
        </TabsContent>
      </Tabs>

      <BottomNav />
    </div>
  )
}
