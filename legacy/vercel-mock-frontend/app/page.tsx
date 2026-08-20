"use client"

import { BottomNav } from "@/components/bottom-nav"
import { PostCard } from "@/components/feed/post-card"
import { FullScreenPostView } from "@/components/feed/full-screen-post-view" // Added full-screen view
import { Button } from "@/components/ui/button"
import { PWAInstallPrompt } from "@/components/pwa-install-prompt"
import { Sparkles } from "lucide-react"
import { useState } from "react"
import type { Post } from "@/lib/types"

const mockPosts: Post[] = [
  {
    id: "1",
    userId: "user1",
    userName: "Sarah Chen",
    userAvatar: "/woman-and-loyal-companion.png",
    petName: "Luna",
    petAvatar: "/golden-retriever.png",
    mediaUrl: "/golden-retriever.png",
    mediaType: "image",
    caption: "Beautiful morning walk at the park! Luna made three new friends today 🌟",
    location: "Golden Gate Park",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    likes: 42,
    comments: 8,
    isLiked: false,
  },
  {
    id: "2",
    userId: "user2",
    userName: "Mike Torres",
    userAvatar: "/man-with-husky.jpg",
    petName: "Max",
    petAvatar: "/siberian-husky-portrait.png",
    mediaUrl: "/man-with-husky.jpg",
    mediaType: "image",
    caption: "Training session complete! Max is getting so good at recall 🎾",
    location: "Riverside Dog Park",
    timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
    likes: 67,
    comments: 12,
    isLiked: true,
  },
  {
    id: "3",
    userId: "user3",
    userName: "Emma Wilson",
    userAvatar: "/yoga-instructor.png",
    petName: "Bailey",
    petAvatar: "/australian-shepherd-portrait.png",
    mediaUrl: "/australian-shepherd-portrait.png",
    mediaType: "image",
    caption: "Beach day with my best friend! Nothing beats those ocean vibes 🌊",
    location: "Ocean Beach",
    timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
    likes: 89,
    comments: 15,
    isLiked: true,
  },
]

export default function HomePage() {
  const [posts, setPosts] = useState<Post[]>(mockPosts)
  const [fullScreenIndex, setFullScreenIndex] = useState<number | null>(null) // Track full-screen state

  const handleLike = (postId: string) => {
    setPosts((prev) =>
      prev.map((post) =>
        post.id === postId
          ? {
              ...post,
              isLiked: !post.isLiked,
              likes: post.isLiked ? post.likes - 1 : post.likes + 1,
            }
          : post,
      ),
    )
  }

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="flex items-center justify-between h-14 px-4 max-w-lg mx-auto">
          <h1 className="text-xl font-bold">PetPath</h1>
          <Button variant="ghost" size="icon" className="relative">
            <Sparkles className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-primary rounded-full" />
          </Button>
        </div>
      </header>

      {/* Feed */}
      <div className="max-w-lg mx-auto">
        {posts.map((post, index) => (
          <PostCard
            key={post.id}
            post={post}
            onLike={handleLike}
            onMediaClick={() => setFullScreenIndex(index)} // Open full-screen on media click
          />
        ))}
      </div>

      {fullScreenIndex !== null && (
        <FullScreenPostView
          posts={posts}
          initialIndex={fullScreenIndex}
          onClose={() => setFullScreenIndex(null)}
          onLike={handleLike}
        />
      )}

      <BottomNav />
      <PWAInstallPrompt />
    </div>
  )
}
