"use client"

import { useState } from "react"
import { Brain, Plus, Smile, Meh, Frown, Zap, Moon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { BottomNav } from "@/components/bottom-nav"
import type { MoodEntry, EnrichmentActivity, BehaviorLog } from "@/lib/types"

export default function WellnessPage() {
  const [selectedPet, setSelectedPet] = useState("p1")

  // Mock mood entries
  const moodEntries: MoodEntry[] = [
    {
      id: "m1",
      petId: "p1",
      date: "2024-03-15",
      mood: "happy",
      notes: "Great energy after morning walk",
      activities: ["Walk", "Play fetch"],
    },
    {
      id: "m2",
      petId: "p1",
      date: "2024-03-14",
      mood: "energetic",
      notes: "Very playful today",
      activities: ["Dog park", "Training"],
    },
    {
      id: "m3",
      petId: "p1",
      date: "2024-03-13",
      mood: "calm",
      notes: "Relaxed day at home",
      activities: ["Puzzle toy", "Nap"],
    },
  ]

  // Mock enrichment activities
  const enrichmentActivities: EnrichmentActivity[] = [
    {
      id: "e1",
      name: "Puzzle Feeder",
      category: "mental",
      description: "Hide treats in a puzzle toy to stimulate problem-solving",
      duration: 15,
      difficulty: "medium",
    },
    {
      id: "e2",
      name: "Scent Work",
      category: "mental",
      description: "Hide treats around the house for your pet to find",
      duration: 20,
      difficulty: "easy",
    },
    {
      id: "e3",
      name: "Agility Course",
      category: "physical",
      description: "Set up obstacles for your pet to navigate",
      duration: 30,
      difficulty: "hard",
    },
    {
      id: "e4",
      name: "Playdate",
      category: "social",
      description: "Arrange a meetup with other pets",
      duration: 60,
      difficulty: "easy",
    },
  ]

  // Mock behavior logs
  const behaviorLogs: BehaviorLog[] = [
    {
      id: "b1",
      petId: "p1",
      date: "2024-03-15",
      behavior: "Learned new trick",
      severity: "positive",
      context: "Training session",
      notes: "Successfully learned 'spin' command",
    },
    {
      id: "b2",
      petId: "p1",
      date: "2024-03-14",
      behavior: "Barking at doorbell",
      severity: "neutral",
      context: "Delivery arrived",
      notes: "Normal alert behavior",
    },
  ]

  const getMoodIcon = (mood: MoodEntry["mood"]) => {
    switch (mood) {
      case "happy":
        return <Smile className="h-6 w-6 text-green-400" />
      case "calm":
        return <Meh className="h-6 w-6 text-blue-400" />
      case "anxious":
        return <Frown className="h-6 w-6 text-orange-400" />
      case "energetic":
        return <Zap className="h-6 w-6 text-yellow-400" />
      case "tired":
        return <Moon className="h-6 w-6 text-purple-400" />
    }
  }

  const getCategoryColor = (category: EnrichmentActivity["category"]) => {
    switch (category) {
      case "mental":
        return "bg-purple-500/10 text-purple-400 border-purple-500/20"
      case "physical":
        return "bg-green-500/10 text-green-400 border-green-500/20"
      case "social":
        return "bg-blue-500/10 text-blue-400 border-blue-500/20"
    }
  }

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-border/40 bg-background/80 backdrop-blur-xl">
        <div className="flex items-center justify-between px-4 py-4">
          <div className="flex items-center gap-3">
            <Brain className="h-6 w-6 text-accent" />
            <h1 className="text-xl font-bold">Wellness</h1>
          </div>
          <Button size="sm" className="gap-2">
            <Plus className="h-4 w-4" />
            Log Mood
          </Button>
        </div>

        {/* Pet Selector */}
        <div className="px-4 pb-3">
          <div className="flex items-center gap-3 rounded-xl border border-border/40 bg-card/50 p-3">
            <Avatar className="h-10 w-10 border-2 border-border">
              <AvatarImage src="/border-collie.jpg" alt="Charlie" />
              <AvatarFallback>C</AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <p className="font-semibold">Charlie</p>
              <p className="text-xs text-muted-foreground">Border Collie • 3 years</p>
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Mood Tracking */}
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">Mood History</h2>
          {moodEntries.map((entry) => (
            <Card key={entry.id} className="glass p-4">
              <div className="flex items-start gap-3">
                {getMoodIcon(entry.mood)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <p className="font-semibold capitalize">{entry.mood}</p>
                    <p className="text-sm text-muted-foreground">{new Date(entry.date).toLocaleDateString()}</p>
                  </div>
                  {entry.notes && <p className="text-sm text-muted-foreground mb-2">{entry.notes}</p>}
                  <div className="flex flex-wrap gap-1">
                    {entry.activities.map((activity) => (
                      <Badge key={activity} variant="outline" className="text-xs">
                        {activity}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        {/* Enrichment Activities */}
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">Enrichment Activities</h2>
          <div className="grid gap-3">
            {enrichmentActivities.map((activity) => (
              <Card key={activity.id} className="glass p-4">
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="flex-1">
                    <p className="font-semibold">{activity.name}</p>
                    <p className="text-sm text-muted-foreground mt-1">{activity.description}</p>
                  </div>
                  <Badge className={getCategoryColor(activity.category)} variant="outline">
                    {activity.category}
                  </Badge>
                </div>
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <span>{activity.duration} min</span>
                  <span>•</span>
                  <span className="capitalize">{activity.difficulty}</span>
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* Behavior Logs */}
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">Behavior Logs</h2>
          {behaviorLogs.map((log) => (
            <Card key={log.id} className="glass p-4">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex-1">
                  <p className="font-semibold">{log.behavior}</p>
                  <p className="text-sm text-muted-foreground">{new Date(log.date).toLocaleDateString()}</p>
                </div>
                <Badge
                  variant="outline"
                  className={
                    log.severity === "positive"
                      ? "bg-green-500/10 text-green-400 border-green-500/20"
                      : log.severity === "concerning"
                        ? "bg-red-500/10 text-red-400 border-red-500/20"
                        : "bg-gray-500/10 text-gray-400 border-gray-500/20"
                  }
                >
                  {log.severity}
                </Badge>
              </div>
              {log.context && <p className="text-sm text-muted-foreground mb-1">Context: {log.context}</p>}
              {log.notes && <p className="text-sm text-muted-foreground">{log.notes}</p>}
            </Card>
          ))}
        </div>
      </div>

      <BottomNav />
    </div>
  )
}
