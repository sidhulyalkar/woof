import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { Friend, FriendRequest } from "@/lib/types"

interface FriendsState {
  friends: Friend[]
  requests: FriendRequest[]
  addFriend: (friend: Friend) => void
  removeFriend: (friendId: string) => void
  acceptRequest: (requestId: string) => void
  declineRequest: (requestId: string) => void
  sendRequest: (userId: string) => void
}

export const useFriendsStore = create<FriendsState>()(
  persist(
    (set) => ({
      friends: [],
      requests: [],
      addFriend: (friend) =>
        set((state) => ({
          friends: [...state.friends, friend],
        })),
      removeFriend: (friendId) =>
        set((state) => ({
          friends: state.friends.filter((f) => f.id !== friendId),
        })),
      acceptRequest: (requestId) =>
        set((state) => {
          const request = state.requests.find((r) => r.id === requestId)
          if (!request) return state

          const newFriend: Friend = {
            id: request.fromUserId,
            name: request.fromUserName,
            avatarUrl: request.fromUserAvatar,
            petName: request.fromPetName,
            petAvatar: request.fromPetAvatar,
            location: "",
            mutualFriends: 0,
            status: "friends",
            friendsSince: new Date().toISOString(),
          }

          return {
            friends: [...state.friends, newFriend],
            requests: state.requests.filter((r) => r.id !== requestId),
          }
        }),
      declineRequest: (requestId) =>
        set((state) => ({
          requests: state.requests.filter((r) => r.id !== requestId),
        })),
      sendRequest: (userId) => {
        console.log(`[v0] Sending friend request to user: ${userId}`)
      },
    }),
    {
      name: "petpath-friends",
    },
  ),
)
