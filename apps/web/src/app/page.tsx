"use client"

import { BottomNav } from "@/components/bottom-nav"
import { PostCard } from "@/components/feed/post-card"
import { FullScreenPostView } from "@/components/feed/full-screen-post-view"
import { Button } from "@/components/ui/button"
import { PWAInstallPrompt } from "@/components/pwa-install-prompt"
import { Sparkles, Loader2 } from "lucide-react"
import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { socialApi } from "@/lib/api"
import type { Post } from "@/lib/types"

export default function HomePage() {
  const queryClient = useQueryClient()
  const [fullScreenIndex, setFullScreenIndex] = useState<number | null>(null)

  // Fetch feed posts
  const { data: posts = [], isLoading, error } = useQuery({
    queryKey: ['feed'],
    queryFn: socialApi.getFeed,
  })

  // Like post mutation
  const likeMutation = useMutation({
    mutationFn: (postId: string) => socialApi.likePost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['feed'] })
    },
  })

  const handleLike = (postId: string) => {
    likeMutation.mutate(postId)
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
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : error ? (
          <div className="text-center py-12 px-4">
            <p className="text-destructive mb-2">Failed to load feed</p>
            <Button variant="outline" onClick={() => queryClient.invalidateQueries({ queryKey: ['feed'] })}>
              Try Again
            </Button>
          </div>
        ) : posts.length === 0 ? (
          <div className="text-center py-12 px-4">
            <p className="text-muted-foreground mb-2">No posts yet</p>
            <p className="text-sm text-muted-foreground">Follow some friends or create your first post!</p>
          </div>
        ) : (
          posts.map((post, index) => (
            <PostCard
              key={post.id}
              post={post}
              onLike={handleLike}
              onMediaClick={() => setFullScreenIndex(index)}
            />
          ))
        )}
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
