"use client"

import { useState } from "react"
import { BottomNav } from "@/components/bottom-nav"
import { ActivityStats } from "@/components/activity/activity-stats"
import { ActivityHistory } from "@/components/activity/activity-history"
import { ActiveWalkCard } from "@/components/activity/active-walk-card"
import { Button } from "@/components/ui/button"
import { Play, TrendingUp } from "lucide-react"
import type { Activity } from "@/lib/types"

// Mock activity data
const mockActivities: Activity[] = [
  {
    id: "a1",
    type: "walk",
    petId: "p1",
    startTime: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
    endTime: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
    duration: 3600,
    distance: 2.5,
    route: [
      { lat: 37.7749, lng: -122.4194 },
      { lat: 37.7759, lng: -122.4184 },
      { lat: 37.7769, lng: -122.4174 },
    ],
    notes: "Great walk in the park! Max loved it.",
  },
  {
    id: "a2",
    type: "walk",
    petId: "p1",
    startTime: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    endTime: new Date(Date.now() - 1000 * 60 * 60 * 24 + 1000 * 60 * 45).toISOString(),
    duration: 2700,
    distance: 1.8,
    route: [
      { lat: 37.7749, lng: -122.4194 },
      { lat: 37.7739, lng: -122.4204 },
    ],
    notes: "Morning walk around the neighborhood.",
  },
  {
    id: "a3",
    type: "walk",
    petId: "p1",
    startTime: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
    endTime: new Date(Date.now() - 1000 * 60 * 60 * 48 + 1000 * 60 * 30).toISOString(),
    duration: 1800,
    distance: 1.2,
    route: [
      { lat: 37.7749, lng: -122.4194 },
      { lat: 37.7744, lng: -122.4199 },
    ],
    notes: "Quick evening walk.",
  },
  {
    id: "a4",
    type: "playdate",
    petId: "p1",
    startTime: new Date(Date.now() - 1000 * 60 * 60 * 72).toISOString(),
    endTime: new Date(Date.now() - 1000 * 60 * 60 * 72 + 1000 * 60 * 90).toISOString(),
    duration: 5400,
    participants: ["o1", "o2"],
    notes: "Playdate with Max and Luna at the dog park.",
  },
]

export default function ActivityPage() {
  const [activities] = useState<Activity[]>(mockActivities)
  const [activeWalk, setActiveWalk] = useState<boolean>(false)

  // Calculate stats
  const thisWeekActivities = activities.filter(
    (a) => new Date(a.startTime).getTime() > Date.now() - 1000 * 60 * 60 * 24 * 7,
  )
  const totalDistance = thisWeekActivities.reduce((sum, a) => sum + (a.distance || 0), 0)
  const totalDuration = thisWeekActivities.reduce((sum, a) => sum + a.duration, 0)
  const totalWalks = thisWeekActivities.filter((a) => a.type === "walk").length

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="px-4 py-4 max-w-lg mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Activity</h1>
              <p className="text-sm text-muted-foreground">Track your pet's activities</p>
            </div>
            <Button size="icon" className="shrink-0">
              <TrendingUp className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="px-4 py-6 max-w-lg mx-auto space-y-6">
        {/* Active Walk */}
        {activeWalk ? (
          <ActiveWalkCard onEnd={() => setActiveWalk(false)} />
        ) : (
          <Button size="lg" className="w-full gap-2" onClick={() => setActiveWalk(true)}>
            <Play className="w-5 h-5" />
            Start Walk
          </Button>
        )}

        {/* Stats */}
        <ActivityStats totalDistance={totalDistance} totalDuration={totalDuration} totalWalks={totalWalks} />

        {/* History */}
        <ActivityHistory activities={activities} />
      </main>

      <BottomNav />
    </div>
  )
}
