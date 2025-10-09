"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { MapPin, Clock, Pause, Play, Square } from "lucide-react"

interface ActiveWalkCardProps {
  onEnd: () => void
}

export function ActiveWalkCard({ onEnd }: ActiveWalkCardProps) {
  const [isPaused, setIsPaused] = useState(false)
  const [duration, setDuration] = useState(0)
  const [distance, setDistance] = useState(0)

  useEffect(() => {
    if (isPaused) return

    const interval = setInterval(() => {
      setDuration((prev) => prev + 1)
      // Simulate distance increase (roughly 3 mph)
      setDistance((prev) => prev + 0.0008)
    }, 1000)

    return () => clearInterval(interval)
  }, [isPaused])

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
    }
    return `${minutes}:${secs.toString().padStart(2, "0")}`
  }

  return (
    <Card className="glass border-primary/50 p-6 space-y-6">
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          {isPaused ? "Paused" : "Walk in Progress"}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center space-y-1">
          <Clock className="w-6 h-6 text-primary mx-auto" />
          <p className="text-3xl font-bold">{formatTime(duration)}</p>
          <p className="text-xs text-muted-foreground">Duration</p>
        </div>

        <div className="text-center space-y-1">
          <MapPin className="w-6 h-6 text-primary mx-auto" />
          <p className="text-3xl font-bold">{distance.toFixed(2)}</p>
          <p className="text-xs text-muted-foreground">Miles</p>
        </div>
      </div>

      {/* Map Placeholder */}
      <div className="relative h-48 rounded-lg overflow-hidden bg-gradient-to-br from-card via-background to-card">
        <div className="absolute inset-0 opacity-10">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="walk-grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#walk-grid)" />
          </svg>
        </div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <MapPin className="w-8 h-8 text-primary mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">Tracking your route...</p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-3">
        <Button
          variant="outline"
          size="lg"
          className="flex-1 gap-2 bg-transparent"
          onClick={() => setIsPaused(!isPaused)}
        >
          {isPaused ? (
            <>
              <Play className="w-5 h-5" />
              Resume
            </>
          ) : (
            <>
              <Pause className="w-5 h-5" />
              Pause
            </>
          )}
        </Button>
        <Button variant="destructive" size="lg" className="flex-1 gap-2" onClick={onEnd}>
          <Square className="w-5 h-5" />
          End Walk
        </Button>
      </div>
    </Card>
  )
}
