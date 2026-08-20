"use client"

import { useState, useRef, useEffect } from "react"
import { X, Volume2, VolumeX, MoreVertical, Heart, MessageCircle, Send } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Progress } from "@/components/ui/progress"
import type { Highlight } from "@/lib/types"

export default function HighlightsPage() {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isMuted, setIsMuted] = useState(false)
  const [progress, setProgress] = useState(0)
  const videoRef = useRef<HTMLVideoElement>(null)

  // Mock highlights data
  const highlights: Highlight[] = [
    {
      id: "h1",
      userId: "1",
      userName: "Sarah Chen",
      userAvatar: "/user-avatar.jpg",
      petName: "Luna",
      petAvatar: "/border-collie.jpg",
      videoUrl: "/dog-fetching-ball.png",
      thumbnail: "/border-collie.jpg",
      caption: "Luna's first time catching a frisbee!",
      timestamp: "2024-03-15T10:30:00Z",
      views: 234,
      expiresAt: "2024-03-16T10:30:00Z",
    },
    {
      id: "h2",
      userId: "2",
      userName: "Mike Rodriguez",
      userAvatar: "/man-with-husky.jpg",
      petName: "Max",
      petAvatar: "/siberian-husky-portrait.png",
      videoUrl: "/husky-in-snow.jpg",
      thumbnail: "/siberian-husky-portrait.png",
      caption: "Snow day adventures!",
      timestamp: "2024-03-15T14:20:00Z",
      views: 189,
      expiresAt: "2024-03-16T14:20:00Z",
    },
  ]

  const currentHighlight = highlights[currentIndex]

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          // Move to next highlight
          if (currentIndex < highlights.length - 1) {
            setCurrentIndex(currentIndex + 1)
            return 0
          }
          return 100
        }
        return prev + 1
      })
    }, 100) // 10 seconds total (100 * 100ms)

    return () => clearInterval(timer)
  }, [currentIndex, highlights.length])

  const handleNext = () => {
    if (currentIndex < highlights.length - 1) {
      setCurrentIndex(currentIndex + 1)
      setProgress(0)
    }
  }

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1)
      setProgress(0)
    }
  }

  const handleClose = () => {
    window.history.back()
  }

  return (
    <div className="fixed inset-0 z-50 bg-black">
      {/* Progress Bars */}
      <div className="absolute top-0 left-0 right-0 z-20 flex gap-1 p-2">
        {highlights.map((_, index) => (
          <div key={index} className="flex-1">
            <Progress
              value={index === currentIndex ? progress : index < currentIndex ? 100 : 0}
              className="h-1 bg-white/30"
            />
          </div>
        ))}
      </div>

      {/* Header */}
      <div className="absolute top-12 left-0 right-0 z-20 flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Avatar className="h-10 w-10 border-2 border-white">
              <AvatarImage src={currentHighlight.userAvatar || "/placeholder.svg"} alt={currentHighlight.userName} />
              <AvatarFallback>{currentHighlight.userName[0]}</AvatarFallback>
            </Avatar>
            <Avatar className="absolute -bottom-1 -right-1 h-5 w-5 border-2 border-black">
              <AvatarImage src={currentHighlight.petAvatar || "/placeholder.svg"} alt={currentHighlight.petName} />
              <AvatarFallback>{currentHighlight.petName[0]}</AvatarFallback>
            </Avatar>
          </div>
          <div>
            <p className="text-sm font-semibold text-white">{currentHighlight.userName}</p>
            <p className="text-xs text-white/80">
              {new Date(currentHighlight.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="icon"
            variant="ghost"
            className="h-8 w-8 text-white hover:bg-white/20"
            onClick={() => setIsMuted(!isMuted)}
          >
            {isMuted ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
          </Button>
          <Button size="icon" variant="ghost" className="h-8 w-8 text-white hover:bg-white/20">
            <MoreVertical className="h-5 w-5" />
          </Button>
          <Button size="icon" variant="ghost" className="h-8 w-8 text-white hover:bg-white/20" onClick={handleClose}>
            <X className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Video Content */}
      <div className="relative h-full w-full">
        {/* Touch areas for navigation */}
        <div className="absolute inset-0 flex">
          <div className="flex-1" onClick={handlePrevious} />
          <div className="flex-1" onClick={handleNext} />
        </div>

        {/* Video placeholder */}
        <div className="flex h-full items-center justify-center bg-gradient-to-br from-primary/20 to-secondary/20">
          <img
            src={currentHighlight.thumbnail || "/placeholder.svg"}
            alt={currentHighlight.caption || "Highlight"}
            className="h-full w-full object-cover"
          />
        </div>
      </div>

      {/* Caption */}
      {currentHighlight.caption && (
        <div className="absolute bottom-32 left-0 right-0 z-20 px-4">
          <p className="text-sm text-white drop-shadow-lg">{currentHighlight.caption}</p>
        </div>
      )}

      {/* Actions */}
      <div className="absolute bottom-20 right-4 z-20 flex flex-col items-center gap-6">
        <button className="flex flex-col items-center gap-1">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm">
            <Heart className="h-6 w-6 text-white" />
          </div>
          <span className="text-xs font-semibold text-white drop-shadow-lg">234</span>
        </button>

        <button className="flex flex-col items-center gap-1">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm">
            <MessageCircle className="h-6 w-6 text-white" />
          </div>
          <span className="text-xs font-semibold text-white drop-shadow-lg">12</span>
        </button>

        <button className="flex flex-col items-center gap-1">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm">
            <Send className="h-6 w-6 text-white" />
          </div>
        </button>
      </div>
    </div>
  )
}
