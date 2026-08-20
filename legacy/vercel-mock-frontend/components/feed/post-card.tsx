"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Heart, MessageCircle, Send, Bookmark, MapPin } from "lucide-react"
import { formatDistanceToNow } from "date-fns"
import type { Post } from "@/lib/types"
import { cn } from "@/lib/utils"

interface PostCardProps {
  post: Post
  onLike: (postId: string) => void
  onMediaClick?: () => void // Added callback for media click
}

export function PostCard({ post, onLike, onMediaClick }: PostCardProps) {
  return (
    <article className="border-b border-border/50">
      {/* Header */}
      <div className="flex items-center gap-3 p-4">
        <Avatar className="w-10 h-10 border-2 border-primary/20">
          <AvatarImage src={post.userAvatar || "/placeholder.svg"} alt={post.userName} />
          <AvatarFallback>{post.userName[0]}</AvatarFallback>
        </Avatar>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className="font-semibold text-sm truncate">{post.userName}</p>
            <span className="text-xs text-muted-foreground">•</span>
            <p className="text-xs text-primary truncate">{post.petName}</p>
          </div>
          {post.location && (
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <MapPin className="w-3 h-3" />
              <span className="truncate">{post.location}</span>
            </div>
          )}
        </div>
        <Button variant="ghost" size="sm" className="text-xs">
          Follow
        </Button>
      </div>

      {/* Media */}
      <div
        className="relative aspect-square bg-muted cursor-pointer hover:opacity-95 transition-opacity"
        onClick={onMediaClick}
      >
        {post.mediaType === "image" ? (
          <img src={post.mediaUrl || "/placeholder.svg"} alt={post.caption} className="w-full h-full object-cover" />
        ) : (
          <video src={post.mediaUrl} className="w-full h-full object-cover" controls playsInline />
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-4 p-4">
        <Button
          variant="ghost"
          size="icon"
          className={cn("hover:scale-110 transition-transform", post.isLiked && "text-red-500")}
          onClick={() => onLike(post.id)}
        >
          <Heart className={cn("w-6 h-6", post.isLiked && "fill-current")} />
        </Button>
        <Button variant="ghost" size="icon" className="hover:scale-110 transition-transform">
          <MessageCircle className="w-6 h-6" />
        </Button>
        <Button variant="ghost" size="icon" className="hover:scale-110 transition-transform">
          <Send className="w-6 h-6" />
        </Button>
        <Button variant="ghost" size="icon" className="ml-auto hover:scale-110 transition-transform">
          <Bookmark className="w-6 h-6" />
        </Button>
      </div>

      {/* Likes & Caption */}
      <div className="px-4 pb-4 space-y-2">
        <p className="font-semibold text-sm">{post.likes.toLocaleString()} likes</p>
        <p className="text-sm">
          <span className="font-semibold mr-2">{post.userName}</span>
          {post.caption}
        </p>
        {post.comments > 0 && (
          <button className="text-sm text-muted-foreground hover:text-foreground">
            View all {post.comments} comments
          </button>
        )}
        <p className="text-xs text-muted-foreground">
          {formatDistanceToNow(new Date(post.timestamp), { addSuffix: true })}
        </p>
      </div>
    </article>
  )
}
