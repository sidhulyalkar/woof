"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Heart, MessageCircle, Send, Bookmark, MapPin, X } from "lucide-react"
import { formatDistanceToNow } from "date-fns"
import type { Post } from "@/lib/types"
import { cn } from "@/lib/utils"

interface FullScreenPostViewProps {
  posts: Post[]
  initialIndex: number
  onClose: () => void
  onLike: (postId: string) => void
}

export function FullScreenPostView({ posts, initialIndex, onClose, onLike }: FullScreenPostViewProps) {
  const [currentIndex, setCurrentIndex] = useState(initialIndex)
  const [showComments, setShowComments] = useState(false)
  const [touchStart, setTouchStart] = useState<{ x: number; y: number } | null>(null)
  const [touchEnd, setTouchEnd] = useState<{ x: number; y: number } | null>(null)
  const [lastTap, setLastTap] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  const currentPost = posts[currentIndex]
  const minSwipeDistance = 50

  // Handle touch start
  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null)
    setTouchStart({
      x: e.targetTouches[0].clientX,
      y: e.targetTouches[0].clientY,
    })
  }

  // Handle touch move
  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd({
      x: e.targetTouches[0].clientX,
      y: e.targetTouches[0].clientY,
    })
  }

  // Handle touch end
  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return

    const distanceX = touchStart.x - touchEnd.x
    const distanceY = touchStart.y - touchEnd.y
    const isHorizontalSwipe = Math.abs(distanceX) > Math.abs(distanceY)
    const isVerticalSwipe = Math.abs(distanceY) > Math.abs(distanceX)

    // Horizontal swipe (left) - open comments
    if (isHorizontalSwipe && distanceX > minSwipeDistance) {
      setShowComments(true)
    }

    // Vertical swipe up - next post
    if (isVerticalSwipe && distanceY > minSwipeDistance && currentIndex < posts.length - 1) {
      setCurrentIndex((prev) => prev + 1)
    }

    // Vertical swipe down - previous post
    if (isVerticalSwipe && distanceY < -minSwipeDistance && currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1)
    }
  }

  // Handle double tap to like
  const handleTap = () => {
    const now = Date.now()
    const DOUBLE_TAP_DELAY = 300

    if (now - lastTap < DOUBLE_TAP_DELAY) {
      // Double tap detected
      onLike(currentPost.id)
    }
    setLastTap(now)
  }

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
      if (e.key === "ArrowUp" && currentIndex > 0) setCurrentIndex((prev) => prev - 1)
      if (e.key === "ArrowDown" && currentIndex < posts.length - 1) setCurrentIndex((prev) => prev + 1)
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [currentIndex, posts.length, onClose])

  return (
    <div className="fixed inset-0 z-50 bg-background">
      {/* Close button */}
      <Button variant="ghost" size="icon" className="absolute top-4 right-4 z-50 glass-strong" onClick={onClose}>
        <X className="w-5 h-5" />
      </Button>

      {/* Post counter */}
      <div className="absolute top-4 left-4 z-50 glass-strong px-3 py-1 rounded-full text-sm">
        {currentIndex + 1} / {posts.length}
      </div>

      {/* Main content */}
      <div
        ref={containerRef}
        className="h-full flex flex-col"
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
        onClick={handleTap}
      >
        {/* Media */}
        <div className="flex-1 relative bg-black flex items-center justify-center">
          {currentPost.mediaType === "image" ? (
            <img
              src={currentPost.mediaUrl || "/placeholder.svg"}
              alt={currentPost.caption}
              className="max-w-full max-h-full object-contain"
            />
          ) : (
            <video
              src={currentPost.mediaUrl}
              className="max-w-full max-h-full object-contain"
              controls
              playsInline
              autoPlay
              loop
            />
          )}

          {/* Swipe indicators */}
          {currentIndex > 0 && (
            <div className="absolute top-1/2 left-4 -translate-y-1/2 text-white/50 text-xs">↓ Previous</div>
          )}
          {currentIndex < posts.length - 1 && (
            <div className="absolute top-1/2 right-4 -translate-y-1/2 text-white/50 text-xs">↑ Next</div>
          )}
        </div>

        {/* Post info overlay */}
        <div className="glass-strong border-t border-border/50">
          {/* Header */}
          <div className="flex items-center gap-3 p-4">
            <Avatar className="w-10 h-10 border-2 border-primary/20">
              <AvatarImage src={currentPost.userAvatar || "/placeholder.svg"} alt={currentPost.userName} />
              <AvatarFallback>{currentPost.userName[0]}</AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <p className="font-semibold text-sm truncate">{currentPost.userName}</p>
                <span className="text-xs text-muted-foreground">•</span>
                <p className="text-xs text-primary truncate">{currentPost.petName}</p>
              </div>
              {currentPost.location && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <MapPin className="w-3 h-3" />
                  <span className="truncate">{currentPost.location}</span>
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4 px-4 pb-4">
            <Button
              variant="ghost"
              size="icon"
              className={cn("hover:scale-110 transition-transform", currentPost.isLiked && "text-red-500")}
              onClick={(e) => {
                e.stopPropagation()
                onLike(currentPost.id)
              }}
            >
              <Heart className={cn("w-6 h-6", currentPost.isLiked && "fill-current")} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="hover:scale-110 transition-transform"
              onClick={(e) => {
                e.stopPropagation()
                setShowComments(true)
              }}
            >
              <MessageCircle className="w-6 h-6" />
            </Button>
            <Button variant="ghost" size="icon" className="hover:scale-110 transition-transform">
              <Send className="w-6 h-6" />
            </Button>
            <Button variant="ghost" size="icon" className="ml-auto hover:scale-110 transition-transform">
              <Bookmark className="w-6 h-6" />
            </Button>
          </div>

          {/* Caption */}
          <div className="px-4 pb-4 space-y-2">
            <p className="font-semibold text-sm">{currentPost.likes.toLocaleString()} likes</p>
            <p className="text-sm">
              <span className="font-semibold mr-2">{currentPost.userName}</span>
              {currentPost.caption}
            </p>
            <p className="text-xs text-muted-foreground">
              {formatDistanceToNow(new Date(currentPost.timestamp), { addSuffix: true })}
            </p>
          </div>
        </div>
      </div>

      {/* Comments sheet */}
      <Sheet open={showComments} onOpenChange={setShowComments}>
        <SheetContent side="bottom" className="h-[80vh]">
          <SheetHeader>
            <SheetTitle>Comments ({currentPost.comments})</SheetTitle>
          </SheetHeader>
          <div className="mt-6 space-y-4">
            <p className="text-sm text-muted-foreground text-center py-8">Comments feature coming soon...</p>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}
