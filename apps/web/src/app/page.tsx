"use client"

import Link from "next/link"
import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Bell,
  CalendarHeart,
  Compass,
  Footprints,
  Loader2,
  PawPrint,
  Trophy,
} from "lucide-react"
import { BottomNav } from "@/components/bottom-nav"
import { PostCard } from "@/components/feed/post-card"
import { FullScreenPostView } from "@/components/feed/full-screen-post-view"
import { Button } from "@/components/ui/button"
import { PWAInstallPrompt } from "@/components/pwa-install-prompt"
import { webSocialApi } from "@/lib/api/social"

const quickActions = [
  {
    href: "/discover",
    label: "Find a match",
    description: "Compatibility-first discovery",
    icon: Compass,
  },
  {
    href: "/events",
    label: "Plan a meetup",
    description: "See what is happening nearby",
    icon: CalendarHeart,
  },
  {
    href: "/activity",
    label: "Log activity",
    description: "Walks, play, runs and hikes",
    icon: Footprints,
  },
  {
    href: "/leaderboard",
    label: "View progress",
    description: "Goals, points and community",
    icon: Trophy,
  },
]

export default function HomePage() {
  const queryClient = useQueryClient()
  const [fullScreenIndex, setFullScreenIndex] = useState<number | null>(null)

  const {
    data: posts = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["feed"],
    queryFn: webSocialApi.getFeed,
  })

  const likeMutation = useMutation({
    mutationFn: ({ postId, isLiked }: { postId: string; isLiked: boolean }) =>
      isLiked ? webSocialApi.unlikePost(postId) : webSocialApi.likePost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["feed"] })
    },
  })

  const handleLike = (postId: string) => {
    const post = posts.find((candidate) => candidate.id === postId)
    if (!post) return
    likeMutation.mutate({ postId, isLiked: post.isLiked })
  }

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <Link href="/" className="flex min-h-0 min-w-0 items-center gap-3" aria-label="Woof home">
            <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
              <PawPrint className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
            </span>
            <span>
              <span className="block text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                Your local pack
              </span>
              <span className="block text-lg font-bold tracking-tight">Woof</span>
            </span>
          </Link>

          <Button variant="ghost" size="icon" asChild className="relative rounded-xl">
            <Link href="/notifications" aria-label="Open notifications">
              <Bell className="h-5 w-5" aria-hidden="true" />
              <span className="absolute right-2 top-2 h-2 w-2 rounded-full border-2 border-background bg-accent" />
            </Link>
          </Button>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-6 pt-5">
        <section aria-labelledby="today-heading" className="animate-in">
          <div className="mb-4 flex items-end justify-between gap-4">
            <div>
              <p className="eyebrow">Start with intent</p>
              <h1 id="today-heading" className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl">
                What does your dog need today?
              </h1>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {quickActions.map((action) => {
              const Icon = action.icon
              return (
                <Link
                  key={action.href}
                  href={action.href}
                  className="group surface-soft flex min-h-[122px] flex-col justify-between rounded-2xl p-4 transition-colors hover:border-primary/30 hover:bg-primary/[0.045]"
                >
                  <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary transition-transform group-hover:-translate-y-0.5">
                    <Icon className="h-5 w-5" aria-hidden="true" />
                  </span>
                  <span className="mt-5">
                    <span className="block text-sm font-semibold">{action.label}</span>
                    <span className="mt-1 block text-xs leading-relaxed text-muted-foreground">
                      {action.description}
                    </span>
                  </span>
                </Link>
              )
            })}
          </div>
        </section>

        <section aria-labelledby="feed-heading" className="mt-8">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <p className="eyebrow">Community</p>
              <h2 id="feed-heading" className="mt-1 text-xl font-bold tracking-tight">
                From your pack
              </h2>
            </div>
            <Link
              href="/discover"
              className="flex min-h-0 min-w-0 items-center gap-1 text-sm font-semibold text-primary hover:text-primary/80"
            >
              Discover
              <span aria-hidden="true">→</span>
            </Link>
          </div>

          <div className="overflow-hidden rounded-2xl border border-border/60 bg-card/55">
            {isLoading ? (
              <div className="flex min-h-52 flex-col items-center justify-center gap-3 px-5 py-12" role="status">
                <Loader2 className="h-7 w-7 animate-spin text-primary" aria-hidden="true" />
                <p className="text-sm text-muted-foreground">Gathering the latest from your pack…</p>
              </div>
            ) : error ? (
              <div className="flex min-h-52 flex-col items-center justify-center px-6 py-12 text-center">
                <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-2xl bg-destructive/10 text-destructive">
                  <PawPrint className="h-5 w-5" aria-hidden="true" />
                </div>
                <h3 className="font-semibold">The feed could not load</h3>
                <p className="mt-1 max-w-xs text-sm leading-relaxed text-muted-foreground">
                  Your profile and other Woof features are still available. Try the feed again when the connection settles.
                </p>
                <Button
                  variant="outline"
                  className="mt-5 bg-transparent"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["feed"] })}
                >
                  Try again
                </Button>
              </div>
            ) : posts.length === 0 ? (
              <div className="flex min-h-56 flex-col items-center justify-center px-6 py-12 text-center">
                <div className="brand-mark mb-4 flex h-12 w-12 items-center justify-center rounded-2xl">
                  <PawPrint className="h-6 w-6 text-primary-foreground" aria-hidden="true" />
                </div>
                <h3 className="text-base font-semibold">Your pack is quiet for now</h3>
                <p className="mt-1 max-w-xs text-sm leading-relaxed text-muted-foreground">
                  Find compatible dogs nearby or share the first walk, play session or park moment.
                </p>
                <div className="mt-5 flex flex-wrap justify-center gap-2">
                  <Button asChild>
                    <Link href="/discover">Find matches</Link>
                  </Button>
                  <Button variant="outline" asChild className="bg-transparent">
                    <Link href="/camera">Create a post</Link>
                  </Button>
                </div>
              </div>
            ) : (
              <div className="divide-y divide-border/50">
                {posts.map((post, index) => (
                  <PostCard
                    key={post.id}
                    post={post}
                    onLike={handleLike}
                    onMediaClick={post.mediaUrl ? () => setFullScreenIndex(index) : undefined}
                  />
                ))}
              </div>
            )}
          </div>
        </section>
      </main>

      {fullScreenIndex !== null && (
        <FullScreenPostView
          posts={posts.filter((post) => Boolean(post.mediaUrl))}
          initialIndex={Math.max(0, posts.slice(0, fullScreenIndex).filter((post) => Boolean(post.mediaUrl)).length)}
          onClose={() => setFullScreenIndex(null)}
          onLike={handleLike}
        />
      )}

      <BottomNav />
      <PWAInstallPrompt />
    </div>
  )
}
