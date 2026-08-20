"use client"

import { useState } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Search, UserMinus, MessageCircle } from "lucide-react"

// Mock data for demonstration
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

const mockFollowers = [
  {
    id: "fo1",
    name: "David Kim",
    avatarUrl: "/man-and-loyal-companion.png",
    petName: "Rocky",
    location: "4.2 miles away",
  },
  {
    id: "fo2",
    name: "Lisa Wang",
    avatarUrl: "/woman-and-cat.png",
    petName: "Whiskers",
    location: "5.1 miles away",
  },
]

const mockFollowing = [
  {
    id: "fw1",
    name: "James Brown",
    avatarUrl: "/man-with-golden-retriever.jpg",
    petName: "Buddy",
    location: "3.5 miles away",
  },
]

export function FriendsListView() {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState("all")

  const filteredFriends = mockFriends.filter(
    (friend) =>
      friend.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      friend.petName.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const filteredFollowers = mockFollowers.filter(
    (follower) =>
      follower.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      follower.petName.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const filteredFollowing = mockFollowing.filter(
    (following) =>
      following.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      following.petName.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <div className="space-y-4 p-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          placeholder="Search friends..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-9"
        />
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="w-full">
          <TabsTrigger value="all" className="flex-1">
            All ({mockFriends.length})
          </TabsTrigger>
          <TabsTrigger value="followers" className="flex-1">
            Followers ({mockFollowers.length})
          </TabsTrigger>
          <TabsTrigger value="following" className="flex-1">
            Following ({mockFollowing.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-2 mt-4">
          {filteredFriends.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <p>No friends found</p>
            </div>
          ) : (
            filteredFriends.map((friend) => (
              <div key={friend.id} className="card p-4 flex items-center gap-3">
                <Avatar className="w-14 h-14">
                  <AvatarImage src={friend.avatarUrl || "/placeholder.svg"} />
                  <AvatarFallback>{friend.name[0]}</AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold truncate">
                    {friend.name} & {friend.petName}
                  </h3>
                  <p className="text-sm text-muted-foreground">{friend.location}</p>
                  <p className="text-xs text-muted-foreground">{friend.mutualFriends} mutual friends</p>
                </div>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline">
                    <MessageCircle className="w-4 h-4" />
                  </Button>
                  <Button size="sm" variant="ghost">
                    <UserMinus className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </TabsContent>

        <TabsContent value="followers" className="space-y-2 mt-4">
          {filteredFollowers.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <p>No followers found</p>
            </div>
          ) : (
            filteredFollowers.map((follower) => (
              <div key={follower.id} className="card p-4 flex items-center gap-3">
                <Avatar className="w-14 h-14">
                  <AvatarImage src={follower.avatarUrl || "/placeholder.svg"} />
                  <AvatarFallback>{follower.name[0]}</AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold truncate">
                    {follower.name} & {follower.petName}
                  </h3>
                  <p className="text-sm text-muted-foreground">{follower.location}</p>
                </div>
                <Button size="sm">Follow Back</Button>
              </div>
            ))
          )}
        </TabsContent>

        <TabsContent value="following" className="space-y-2 mt-4">
          {filteredFollowing.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <p>Not following anyone</p>
            </div>
          ) : (
            filteredFollowing.map((following) => (
              <div key={following.id} className="card p-4 flex items-center gap-3">
                <Avatar className="w-14 h-14">
                  <AvatarImage src={following.avatarUrl || "/placeholder.svg"} />
                  <AvatarFallback>{following.name[0]}</AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold truncate">
                    {following.name} & {following.petName}
                  </h3>
                  <p className="text-sm text-muted-foreground">{following.location}</p>
                </div>
                <Button size="sm" variant="outline">
                  Unfollow
                </Button>
              </div>
            ))
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
