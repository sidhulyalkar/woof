"use client"

import { Trophy } from "lucide-react"
import { Progress } from "@/components/ui/progress"

interface LevelProgressProps {
  level: number
  points: number
  className?: string
}

export function LevelProgress({ level, points, className }: LevelProgressProps) {
  const pointsForCurrentLevel = (level - 1) * 1000
  const pointsForNextLevel = level * 1000
  const progressInLevel = points - pointsForCurrentLevel
  const progressPercentage = (progressInLevel / 1000) * 100

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Trophy className="h-4 w-4 text-accent" />
          <span className="text-sm font-semibold">Level {level}</span>
        </div>
        <span className="text-xs text-muted-foreground">
          {progressInLevel}/{1000} XP
        </span>
      </div>
      <Progress value={progressPercentage} className="h-2" />
    </div>
  )
}
