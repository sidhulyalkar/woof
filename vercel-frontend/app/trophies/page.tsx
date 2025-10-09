"use client"

import { useState } from "react"
import { BottomNav } from "@/components/bottom-nav"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Progress } from "@/components/ui/progress"
import { Trophy, Award, Lock, Sparkles, Users, Activity, Heart, Star, Crown, TrendingUp } from "lucide-react"
import type { Badge as BadgeType } from "@/lib/types"
import { cn } from "@/lib/utils"

// Mock badge data
const earnedBadges: BadgeType[] = [
  {
    id: "b1",
    name: "Early Bird",
    description: "Complete 10 morning walks before 8 AM",
    iconUrl: "/badge-early-bird.png",
    category: "activity",
    rarity: "rare",
    unlockedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(),
  },
  {
    id: "b2",
    name: "Social Butterfly",
    description: "Attend 5 playdates with other pet parents",
    iconUrl: "/badge-social.png",
    category: "social",
    rarity: "common",
    unlockedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 14).toISOString(),
  },
  {
    id: "b3",
    name: "Marathon Walker",
    description: "Walk 100 miles total",
    iconUrl: "/badge-marathon.png",
    category: "activity",
    rarity: "epic",
    unlockedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3).toISOString(),
  },
  {
    id: "b4",
    name: "Health Champion",
    description: "Complete all wellness checkups for 6 months",
    iconUrl: "/badge-health.png",
    category: "health",
    rarity: "rare",
    unlockedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(),
  },
  {
    id: "b5",
    name: "Community Leader",
    description: "Organize 3 community events",
    iconUrl: "/badge-leader.png",
    category: "special",
    rarity: "legendary",
    unlockedAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
  },
  {
    id: "b6",
    name: "Streak Master",
    description: "Maintain a 30-day activity streak",
    iconUrl: "/badge-streak.png",
    category: "activity",
    rarity: "epic",
    unlockedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 5).toISOString(),
  },
]

const lockedBadges = [
  {
    id: "l1",
    name: "Century Club",
    description: "Walk 500 miles total",
    category: "activity" as const,
    rarity: "legendary" as const,
    progress: 65,
    requirement: "Walk 500 miles (325/500)",
  },
  {
    id: "l2",
    name: "Party Animal",
    description: "Attend 20 playdates",
    category: "social" as const,
    rarity: "rare" as const,
    progress: 40,
    requirement: "Attend 20 playdates (8/20)",
  },
  {
    id: "l3",
    name: "Wellness Guru",
    description: "Log 100 wellness activities",
    category: "health" as const,
    rarity: "epic" as const,
    progress: 78,
    requirement: "Log 100 activities (78/100)",
  },
  {
    id: "l4",
    name: "Speed Demon",
    description: "Complete a 5-mile walk in under 60 minutes",
    category: "activity" as const,
    rarity: "rare" as const,
    progress: 20,
    requirement: "Best time: 75 minutes",
  },
  {
    id: "l5",
    name: "Influencer",
    description: "Get 1000 likes on your posts",
    category: "social" as const,
    rarity: "epic" as const,
    progress: 55,
    requirement: "Get 1000 likes (550/1000)",
  },
  {
    id: "l6",
    name: "Perfect Week",
    description: "Complete all daily goals for 7 days straight",
    category: "special" as const,
    rarity: "legendary" as const,
    progress: 85,
    requirement: "Complete 7 perfect days (6/7)",
  },
]

const categoryIcons = {
  social: Users,
  activity: Activity,
  health: Heart,
  special: Star,
}

const rarityColors = {
  common: "from-gray-400 to-gray-500",
  rare: "from-blue-400 to-cyan-500",
  epic: "from-purple-400 to-pink-500",
  legendary: "from-yellow-400 to-orange-500",
}

const rarityBorders = {
  common: "border-gray-400",
  rare: "border-blue-400",
  epic: "border-purple-400",
  legendary: "border-yellow-400",
}

export default function TrophiesPage() {
  const [selectedBadge, setSelectedBadge] = useState<BadgeType | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<BadgeType["category"] | "all">("all")

  const filteredEarned =
    selectedCategory === "all" ? earnedBadges : earnedBadges.filter((b) => b.category === selectedCategory)

  const filteredLocked =
    selectedCategory === "all" ? lockedBadges : lockedBadges.filter((b) => b.category === selectedCategory)

  const stats = {
    total: earnedBadges.length,
    common: earnedBadges.filter((b) => b.rarity === "common").length,
    rare: earnedBadges.filter((b) => b.rarity === "rare").length,
    epic: earnedBadges.filter((b) => b.rarity === "epic").length,
    legendary: earnedBadges.filter((b) => b.rarity === "legendary").length,
  }

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="px-4 py-4 max-w-lg mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold flex items-center gap-2">
                <Trophy className="w-6 h-6 text-primary" />
                Trophy Case
              </h1>
              <p className="text-sm text-muted-foreground">{earnedBadges.length} achievements unlocked</p>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-1 justify-end">
                <Crown className="w-5 h-5 text-yellow-500" />
                <span className="text-2xl font-bold">{stats.legendary}</span>
              </div>
              <p className="text-xs text-muted-foreground">Legendary</p>
            </div>
          </div>
        </div>
      </header>

      {/* Stats Overview */}
      <div className="px-4 py-6 max-w-lg mx-auto space-y-6">
        <div className="grid grid-cols-4 gap-3">
          <Card className="glass p-3 text-center">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-gray-400 to-gray-500 mx-auto mb-2 flex items-center justify-center">
              <Award className="w-4 h-4 text-white" />
            </div>
            <p className="text-lg font-bold">{stats.common}</p>
            <p className="text-xs text-muted-foreground">Common</p>
          </Card>

          <Card className="glass p-3 text-center">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-cyan-500 mx-auto mb-2 flex items-center justify-center">
              <Award className="w-4 h-4 text-white" />
            </div>
            <p className="text-lg font-bold">{stats.rare}</p>
            <p className="text-xs text-muted-foreground">Rare</p>
          </Card>

          <Card className="glass p-3 text-center">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-400 to-pink-500 mx-auto mb-2 flex items-center justify-center">
              <Award className="w-4 h-4 text-white" />
            </div>
            <p className="text-lg font-bold">{stats.epic}</p>
            <p className="text-xs text-muted-foreground">Epic</p>
          </Card>

          <Card className="glass p-3 text-center">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 mx-auto mb-2 flex items-center justify-center">
              <Crown className="w-4 h-4 text-white" />
            </div>
            <p className="text-lg font-bold">{stats.legendary}</p>
            <p className="text-xs text-muted-foreground">Legendary</p>
          </Card>
        </div>

        {/* Recent Achievement */}
        <Card className="glass p-4 border-2 border-primary/50">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Latest Achievement</h3>
          </div>
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "w-16 h-16 rounded-full bg-gradient-to-br flex items-center justify-center shrink-0",
                rarityColors[earnedBadges[0].rarity],
              )}
            >
              <Crown className="w-8 h-8 text-white" />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h4 className="font-semibold">{earnedBadges[0].name}</h4>
                <Badge variant="secondary" className="capitalize text-xs">
                  {earnedBadges[0].rarity}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-1">{earnedBadges[0].description}</p>
              <p className="text-xs text-muted-foreground mt-2">
                Unlocked {new Date(earnedBadges[0].unlockedAt!).toLocaleDateString()}
              </p>
            </div>
          </div>
        </Card>

        {/* Category Filter */}
        <div className="flex gap-2 overflow-x-auto pb-2">
          <Button
            variant={selectedCategory === "all" ? "default" : "outline"}
            size="sm"
            onClick={() => setSelectedCategory("all")}
            className={cn(!selectedCategory && "bg-transparent")}
          >
            <Trophy className="w-4 h-4 mr-1" />
            All
          </Button>
          {(Object.keys(categoryIcons) as BadgeType["category"][]).map((category) => {
            const Icon = categoryIcons[category]
            return (
              <Button
                key={category}
                variant={selectedCategory === category ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedCategory(category)}
                className={cn(selectedCategory !== category && "bg-transparent", "capitalize shrink-0")}
              >
                <Icon className="w-4 h-4 mr-1" />
                {category}
              </Button>
            )
          })}
        </div>

        {/* Tabs */}
        <Tabs defaultValue="earned" className="w-full">
          <TabsList className="grid w-full grid-cols-2 bg-muted/50">
            <TabsTrigger value="earned">Earned ({filteredEarned.length})</TabsTrigger>
            <TabsTrigger value="locked">Locked ({filteredLocked.length})</TabsTrigger>
          </TabsList>

          {/* Earned Badges */}
          <TabsContent value="earned" className="mt-4 space-y-3">
            {filteredEarned.map((badge) => {
              const Icon = categoryIcons[badge.category]
              return (
                <Card
                  key={badge.id}
                  className={cn(
                    "glass p-4 cursor-pointer hover:bg-accent/50 transition-colors border-2",
                    rarityBorders[badge.rarity],
                  )}
                  onClick={() => setSelectedBadge(badge)}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={cn(
                        "w-14 h-14 rounded-full bg-gradient-to-br flex items-center justify-center shrink-0",
                        rarityColors[badge.rarity],
                      )}
                    >
                      <Icon className="w-7 h-7 text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h4 className="font-semibold truncate">{badge.name}</h4>
                        <Badge variant="secondary" className="capitalize text-xs shrink-0">
                          {badge.rarity}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{badge.description}</p>
                      <p className="text-xs text-muted-foreground mt-2">
                        {new Date(badge.unlockedAt!).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </Card>
              )
            })}
          </TabsContent>

          {/* Locked Badges */}
          <TabsContent value="locked" className="mt-4 space-y-3">
            {filteredLocked.map((badge) => {
              const Icon = categoryIcons[badge.category]
              return (
                <Card key={badge.id} className="glass p-4 opacity-75">
                  <div className="flex items-center gap-3">
                    <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center shrink-0 relative">
                      <Icon className="w-7 h-7 text-muted-foreground" />
                      <div className="absolute inset-0 flex items-center justify-center bg-background/50 rounded-full">
                        <Lock className="w-5 h-5 text-muted-foreground" />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h4 className="font-semibold truncate">{badge.name}</h4>
                        <Badge variant="outline" className="capitalize text-xs shrink-0">
                          {badge.rarity}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{badge.description}</p>
                      <div className="mt-3 space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">{badge.requirement}</span>
                          <span className="font-semibold">{badge.progress}%</span>
                        </div>
                        <Progress value={badge.progress} className="h-1.5" />
                      </div>
                    </div>
                  </div>
                </Card>
              )
            })}
          </TabsContent>
        </Tabs>

        {/* Achievement Stats */}
        <Card className="glass p-4">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Achievement Stats</h3>
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Completion Rate</span>
              <span className="font-semibold">
                {Math.round((earnedBadges.length / (earnedBadges.length + lockedBadges.length)) * 100)}%
              </span>
            </div>
            <Progress
              value={(earnedBadges.length / (earnedBadges.length + lockedBadges.length)) * 100}
              className="h-2"
            />
            <div className="grid grid-cols-2 gap-3 pt-2">
              <div>
                <p className="text-xs text-muted-foreground">Total Unlocked</p>
                <p className="text-xl font-bold">{earnedBadges.length}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">In Progress</p>
                <p className="text-xl font-bold">{lockedBadges.length}</p>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Badge Detail Sheet */}
      <Sheet open={!!selectedBadge} onOpenChange={() => setSelectedBadge(null)}>
        <SheetContent side="bottom" className="h-[70vh]">
          {selectedBadge && (
            <>
              <SheetHeader>
                <SheetTitle>Achievement Details</SheetTitle>
              </SheetHeader>
              <div className="mt-6 space-y-6">
                {/* Badge Display */}
                <div className="flex flex-col items-center text-center">
                  <div
                    className={cn(
                      "w-24 h-24 rounded-full bg-gradient-to-br flex items-center justify-center mb-4",
                      rarityColors[selectedBadge.rarity],
                    )}
                  >
                    {categoryIcons[selectedBadge.category] &&
                      (() => {
                        const Icon = categoryIcons[selectedBadge.category]
                        return <Icon className="w-12 h-12 text-white" />
                      })()}
                  </div>
                  <h3 className="text-2xl font-bold">{selectedBadge.name}</h3>
                  <Badge variant="secondary" className="capitalize mt-2">
                    {selectedBadge.rarity}
                  </Badge>
                </div>

                {/* Description */}
                <div>
                  <h4 className="font-semibold mb-2">Description</h4>
                  <p className="text-sm text-muted-foreground">{selectedBadge.description}</p>
                </div>

                {/* Details */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between py-2 border-b border-border/50">
                    <span className="text-sm text-muted-foreground">Category</span>
                    <Badge variant="outline" className="capitalize">
                      {selectedBadge.category}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between py-2 border-b border-border/50">
                    <span className="text-sm text-muted-foreground">Rarity</span>
                    <Badge variant="outline" className="capitalize">
                      {selectedBadge.rarity}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between py-2 border-b border-border/50">
                    <span className="text-sm text-muted-foreground">Unlocked</span>
                    <span className="text-sm font-medium">
                      {new Date(selectedBadge.unlockedAt!).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "long",
                        day: "numeric",
                      })}
                    </span>
                  </div>
                </div>

                {/* Share Button */}
                <Button className="w-full gap-2">
                  <Sparkles className="w-4 h-4" />
                  Share Achievement
                </Button>
              </div>
            </>
          )}
        </SheetContent>
      </Sheet>

      <BottomNav />
    </div>
  )
}
