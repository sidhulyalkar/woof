"use client"

import { Heart, MessageCircle } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { AppImage } from "@/components/ui/app-image"
import { Button } from "@/components/ui/button"
import type { Post } from "@/lib/types"
import { cn } from "@/lib/utils"

interface PostCardProps {
  post: Post
  onLike: (postId: string) => void
  onMediaClick?: () => void
}

export function PostCard({ post, onLike, onMediaClick }: PostCardProps) {
  const relativeTime = formatRelativeTime(post.timestamp)

  return (
    <article className="bg-card/20">
      <div className="flex items-center gap-3 px-4 py-3.5">
        <Avatar className="h-10 w-10 border border-border">
          <AvatarImage src={post.userAvatar || "/placeholder.svg"} alt="" />
          <AvatarFallback>{post.userName.slice(0, 1).toUpperCase()}</AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1">
          <p className="truncate font-semibold">@{post.userName}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {post.petName && post.petName !== "Woof" ? `with ${post.petName} · ` : ""}{relativeTime}
          </p>
        </div>
      </div>

      {post.caption && (
        <p className="px-4 pb-3 text-sm leading-6 text-foreground/90">{post.caption}</p>
      )}

      {post.mediaUrl && (
        <button
          type="button"
          onClick={onMediaClick}
          disabled={!onMediaClick}
          className="block min-h-0 w-full min-w-0 overflow-hidden bg-muted/25 text-left disabled:cursor-default"
          aria-label={onMediaClick ? "Open post media" : undefined}
        >
          {post.mediaType === "video" ? (
            <video src={post.mediaUrl} className="max-h-[640px] w-full object-cover" muted playsInline preload="metadata" />
          ) : (
            <AppImage
              src={post.mediaUrl}
              alt=""
              width={1200}
              height={900}
              className="max-h-[640px] w-full object-cover"
              loading="lazy"
            />
          )}
        </button>
      )}

      <div className="flex items-center gap-3 px-3 py-2.5">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className={cn("gap-2 rounded-xl", post.isLiked && "text-accent")}
          onClick={() => onLike(post.id)}
          aria-pressed={post.isLiked}
          aria-label={post.isLiked ? `Unlike post with ${post.likes} likes` : `Like post with ${post.likes} likes`}
        >
          <Heart className={cn("h-5 w-5", post.isLiked && "fill-current")} aria-hidden="true" />
          <span className="tabular-nums">{post.likes}</span>
        </Button>

        <span className="inline-flex items-center gap-2 px-2 text-sm text-muted-foreground" aria-label={`${post.comments} comments`}>
          <MessageCircle className="h-5 w-5" aria-hidden="true" />
          <span className="tabular-nums">{post.comments}</span>
        </span>
      </div>
    </article>
  )
}

function formatRelativeTime(value: string) {
  const timestamp = new Date(value).getTime()
  if (!Number.isFinite(timestamp)) return "Recently"

  const elapsed = Date.now() - timestamp
  const minutes = Math.max(0, Math.floor(elapsed / 60_000))
  if (minutes < 1) return "Just now"
  if (minutes < 60) return `${minutes}m ago`

  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`

  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`

  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(timestamp))
}
