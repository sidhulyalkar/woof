"use client"

import { useState } from "react"
import { BottomNav } from "@/components/bottom-nav"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Trophy,
  TrendingUp,
  TrendingDown,
  Minus,
  Crown,
  Medal,
  Award,
  Zap,
  MapPin,
  Users,
  Activity,
} from "lucide-react"
import { Line, LineChart, Bar, BarChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import type { LeaderboardEntry } from "@/lib/types"
import { cn } from "@/lib/utils"

const TIMEFRAMES = ["daily", "weekly", "monthly", "all-time"] as const

// Mock leaderboard data
const leaderboard: LeaderboardEntry[] = [
  {
    rank: 1,
    userId: "1",
    userName: "Sarah Chen",
    userAvatar: "/user-avatar.jpg",
    petName: "Luna",
    petAvatar: "/border-collie.jpg",
    points: 15420,
    level: 16,
    badges: 24,
    change: 2,
  },
  {
    rank: 2,
    userId: "2",
    userName: "Mike Rodriguez",
    userAvatar: "/man-with-husky.jpg",
    petName: "Max",
    petAvatar: "/siberian-husky-portrait.png",
    points: 14850,
    level: 15,
    badges: 21,
    change: -1,
  },
  {
    rank: 3,
    userId: "3",
    userName: "Emma Wilson",
    userAvatar: "/woman-and-loyal-companion.png",
    petName: "Charlie",
    petAvatar: "/golden-retriever.png",
    points: 13920,
    level: 14,
    badges: 19,
    change: 1,
  },
  {
    rank: 4,
    userId: "4",
    userName: "Alex Kim",
    userAvatar: "/yoga-instructor.png",
    petName: "Bella",
    petAvatar: "/australian-shepherd-portrait.png",
    points: 12340,
    level: 13,
    badges: 17,
    change: 0,
  },
  {
    rank: 5,
    userId: "5",
    userName: "Jordan Lee",
    userAvatar: "/placeholder.svg?height=100&width=100",
    petName: "Rocky",
    petAvatar: "/placeholder.svg?height=100&width=100",
    points: 11890,
    level: 13,
    badges: 16,
    change: 3,
  },
  {
    rank: 6,
    userId: "6",
    userName: "Taylor Swift",
    userAvatar: "/placeholder.svg?height=100&width=100",
    petName: "Meredith",
    petAvatar: "/placeholder.svg?height=100&width=100",
    points: 10450,
    level: 12,
    badges: 15,
    change: -2,
  },
]

// Mock category leaderboards
const distanceLeaderboard = [
  { rank: 1, userName: "Sarah Chen", petName: "Luna", value: 245.8, unit: "mi" },
  { rank: 2, userName: "Mike Rodriguez", petName: "Max", value: 232.4, unit: "mi" },
  { rank: 3, userName: "Emma Wilson", petName: "Charlie", value: 218.9, unit: "mi" },
]

const socialLeaderboard = [
  { rank: 1, userName: "Emma Wilson", petName: "Charlie", value: 156, unit: "friends" },
  { rank: 2, userName: "Sarah Chen", petName: "Luna", value: 142, unit: "friends" },
  { rank: 3, userName: "Alex Kim", petName: "Bella", value: 128, unit: "friends" },
]

const activityLeaderboard = [
  { rank: 1, userName: "Mike Rodriguez", petName: "Max", value: 89, unit: "walks" },
  { rank: 2, userName: "Sarah Chen", petName: "Luna", value: 85, unit: "walks" },
  { rank: 3, userName: "Jordan Lee", petName: "Rocky", value: 78, unit: "walks" },
]

// Mock trend data
const pointsTrend = [
  { day: "Mon", points: 12500 },
  { day: "Tue", points: 13200 },
  { day: "Wed", points: 13800 },
  { day: "Thu", points: 14100 },
  { day: "Fri", points: 14600 },
  { day: "Sat", points: 15100 },
  { day: "Sun", points: 15420 },
]

const topPerformers = [
  { name: "Sarah", points: 15420 },
  { name: "Mike", points: 14850 },
  { name: "Emma", points: 13920 },
  { name: "Alex", points: 12340 },
  { name: "Jordan", points: 11890 },
]

export default function LeaderboardPage() {
  const [timeframe, setTimeframe] = useState<(typeof TIMEFRAMES)[number]>("weekly")
  const currentUserId = "4" // Mock current user

  const getRankIcon = (rank: number) => {
    if (rank === 1) return <Crown className="h-6 w-6 text-yellow-400" />
    if (rank === 2) return <Medal className="h-6 w-6 text-gray-300" />
    if (rank === 3) return <Award className="h-6 w-6 text-amber-600" />
    return <span className="text-lg font-bold text-muted-foreground">{rank}</span>
  }

  const getChangeIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="h-4 w-4 text-green-400" />
    if (change < 0) return <TrendingDown className="h-4 w-4 text-red-400" />
    return <Minus className="h-4 w-4 text-muted-foreground" />
  }

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="px-4 py-4 max-w-lg mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Trophy className="h-6 w-6 text-primary" />
              <div>
                <h1 className="text-2xl font-bold">Leaderboard</h1>
                <p className="text-sm text-muted-foreground capitalize">{timeframe} rankings</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="px-4 py-6 max-w-lg mx-auto space-y-6">
        {/* Timeframe Selector */}
        <div className="flex gap-2 overflow-x-auto pb-2">
          {TIMEFRAMES.map((tf) => (
            <Button
              key={tf}
              variant={timeframe === tf ? "default" : "outline"}
              size="sm"
              onClick={() => setTimeframe(tf)}
              className={cn("capitalize shrink-0", timeframe !== tf && "bg-transparent")}
            >
              {tf}
            </Button>
          ))}
        </div>

        {/* Top 3 Podium */}
        <Card className="glass p-6">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Crown className="w-5 h-5 text-yellow-500" />
            Top Performers
          </h3>
          <div className="flex items-end justify-center gap-4">
            {/* 2nd Place */}
            {leaderboard[1] && (
              <div className="flex flex-col items-center flex-1">
                <Avatar className="w-16 h-16 border-4 border-gray-300 mb-2">
                  <AvatarImage src={leaderboard[1].userAvatar || "/placeholder.svg"} />
                  <AvatarFallback>{leaderboard[1].userName[0]}</AvatarFallback>
                </Avatar>
                <Medal className="w-6 h-6 text-gray-300 mb-1" />
                <p className="text-sm font-semibold text-center truncate w-full">{leaderboard[1].userName}</p>
                <p className="text-xs text-muted-foreground">{leaderboard[1].points.toLocaleString()}</p>
                <div className="w-full h-20 bg-gradient-to-t from-gray-300/20 to-transparent rounded-t-lg mt-2" />
              </div>
            )}

            {/* 1st Place */}
            {leaderboard[0] && (
              <div className="flex flex-col items-center flex-1">
                <Avatar className="w-20 h-20 border-4 border-yellow-400 mb-2">
                  <AvatarImage src={leaderboard[0].userAvatar || "/placeholder.svg"} />
                  <AvatarFallback>{leaderboard[0].userName[0]}</AvatarFallback>
                </Avatar>
                <Crown className="w-7 h-7 text-yellow-400 mb-1" />
                <p className="text-sm font-semibold text-center truncate w-full">{leaderboard[0].userName}</p>
                <p className="text-xs text-muted-foreground">{leaderboard[0].points.toLocaleString()}</p>
                <div className="w-full h-28 bg-gradient-to-t from-yellow-400/20 to-transparent rounded-t-lg mt-2" />
              </div>
            )}

            {/* 3rd Place */}
            {leaderboard[2] && (
              <div className="flex flex-col items-center flex-1">
                <Avatar className="w-16 h-16 border-4 border-amber-600 mb-2">
                  <AvatarImage src={leaderboard[2].userAvatar || "/placeholder.svg"} />
                  <AvatarFallback>{leaderboard[2].userName[0]}</AvatarFallback>
                </Avatar>
                <Award className="w-6 h-6 text-amber-600 mb-1" />
                <p className="text-sm font-semibold text-center truncate w-full">{leaderboard[2].userName}</p>
                <p className="text-xs text-muted-foreground">{leaderboard[2].points.toLocaleString()}</p>
                <div className="w-full h-16 bg-gradient-to-t from-amber-600/20 to-transparent rounded-t-lg mt-2" />
              </div>
            )}
          </div>
        </Card>

        {/* Points Trend Chart */}
        <Card className="glass p-4">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Leader's Progress</h3>
          </div>
          <ChartContainer
            config={{
              points: {
                label: "Points",
                color: "hsl(var(--chart-1))",
              },
            }}
            className="h-[180px]"
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pointsTrend}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="day" fontSize={12} />
                <YAxis fontSize={12} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line type="monotone" dataKey="points" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>

        {/* Category Leaderboards */}
        <Tabs defaultValue="overall" className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-muted/50">
            <TabsTrigger value="overall" className="text-xs">
              Overall
            </TabsTrigger>
            <TabsTrigger value="distance" className="text-xs">
              Distance
            </TabsTrigger>
            <TabsTrigger value="social" className="text-xs">
              Social
            </TabsTrigger>
            <TabsTrigger value="activity" className="text-xs">
              Activity
            </TabsTrigger>
          </TabsList>

          {/* Overall Leaderboard */}
          <TabsContent value="overall" className="mt-4 space-y-2">
            {leaderboard.map((entry) => (
              <Card
                key={entry.userId}
                className={cn(
                  "glass p-4 transition-colors",
                  entry.rank <= 3 && "border-2 border-primary/30",
                  entry.userId === currentUserId && "bg-primary/5 border-2 border-primary",
                )}
              >
                <div className="flex items-center gap-4">
                  {/* Rank */}
                  <div className="flex w-10 items-center justify-center shrink-0">{getRankIcon(entry.rank)}</div>

                  {/* User & Pet Info */}
                  <div className="flex flex-1 items-center gap-3 min-w-0">
                    <div className="relative shrink-0">
                      <Avatar className="h-12 w-12 border-2 border-border">
                        <AvatarImage src={entry.userAvatar || "/placeholder.svg"} alt={entry.userName} />
                        <AvatarFallback>{entry.userName[0]}</AvatarFallback>
                      </Avatar>
                      <Avatar className="absolute -bottom-1 -right-1 h-6 w-6 border-2 border-background">
                        <AvatarImage src={entry.petAvatar || "/placeholder.svg"} alt={entry.petName} />
                        <AvatarFallback>{entry.petName[0]}</AvatarFallback>
                      </Avatar>
                    </div>

                    <div className="flex-1 min-w-0">
                      <p className="font-semibold truncate">{entry.userName}</p>
                      <p className="text-sm text-muted-foreground truncate">with {entry.petName}</p>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="font-mono text-xs">
                        Lv {entry.level}
                      </Badge>
                      {getChangeIcon(entry.change)}
                    </div>
                    <p className="text-sm font-bold text-primary">{entry.points.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">{entry.badges} badges</p>
                  </div>
                </div>
              </Card>
            ))}
          </TabsContent>

          {/* Distance Leaderboard */}
          <TabsContent value="distance" className="mt-4 space-y-3">
            <Card className="glass p-4">
              <div className="flex items-center gap-2 mb-4">
                <MapPin className="w-5 h-5 text-primary" />
                <h3 className="font-semibold">Top Distance Walkers</h3>
              </div>
              <div className="space-y-3">
                {distanceLeaderboard.map((entry) => (
                  <div key={entry.rank} className="flex items-center gap-3">
                    <div className="w-8 text-center shrink-0">{getRankIcon(entry.rank)}</div>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm truncate">{entry.userName}</p>
                      <p className="text-xs text-muted-foreground truncate">{entry.petName}</p>
                    </div>
                    <div className="text-right shrink-0">
                      <p className="font-bold text-primary">{entry.value}</p>
                      <p className="text-xs text-muted-foreground">{entry.unit}</p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>

          {/* Social Leaderboard */}
          <TabsContent value="social" className="mt-4 space-y-3">
            <Card className="glass p-4">
              <div className="flex items-center gap-2 mb-4">
                <Users className="w-5 h-5 text-primary" />
                <h3 className="font-semibold">Most Social</h3>
              </div>
              <div className="space-y-3">
                {socialLeaderboard.map((entry) => (
                  <div key={entry.rank} className="flex items-center gap-3">
                    <div className="w-8 text-center shrink-0">{getRankIcon(entry.rank)}</div>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm truncate">{entry.userName}</p>
                      <p className="text-xs text-muted-foreground truncate">{entry.petName}</p>
                    </div>
                    <div className="text-right shrink-0">
                      <p className="font-bold text-primary">{entry.value}</p>
                      <p className="text-xs text-muted-foreground">{entry.unit}</p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>

          {/* Activity Leaderboard */}
          <TabsContent value="activity" className="mt-4 space-y-3">
            <Card className="glass p-4">
              <div className="flex items-center gap-2 mb-4">
                <Activity className="w-5 h-5 text-primary" />
                <h3 className="font-semibold">Most Active</h3>
              </div>
              <div className="space-y-3">
                {activityLeaderboard.map((entry) => (
                  <div key={entry.rank} className="flex items-center gap-3">
                    <div className="w-8 text-center shrink-0">{getRankIcon(entry.rank)}</div>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm truncate">{entry.userName}</p>
                      <p className="text-xs text-muted-foreground truncate">{entry.petName}</p>
                    </div>
                    <div className="text-right shrink-0">
                      <p className="font-bold text-primary">{entry.value}</p>
                      <p className="text-xs text-muted-foreground">{entry.unit}</p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Points Distribution */}
        <Card className="glass p-4">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Points Distribution</h3>
          </div>
          <ChartContainer
            config={{
              points: {
                label: "Points",
                color: "hsl(var(--chart-2))",
              },
            }}
            className="h-[180px]"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topPerformers}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="name" fontSize={12} />
                <YAxis fontSize={12} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="points" fill="hsl(var(--chart-2))" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>

        {/* Your Position */}
        {leaderboard.find((e) => e.userId === currentUserId) && (
          <Card className="glass p-4 border-2 border-primary">
            <div className="flex items-center gap-2 mb-3">
              <Trophy className="w-5 h-5 text-primary" />
              <h3 className="font-semibold">Your Position</h3>
            </div>
            {(() => {
              const userEntry = leaderboard.find((e) => e.userId === currentUserId)!
              return (
                <div className="flex items-center gap-4">
                  <div className="text-3xl font-bold text-primary">#{userEntry.rank}</div>
                  <div className="flex-1">
                    <p className="font-semibold">{userEntry.userName}</p>
                    <p className="text-sm text-muted-foreground">
                      {userEntry.points.toLocaleString()} points • Level {userEntry.level}
                    </p>
                  </div>
                  <div className="flex items-center gap-1">
                    {getChangeIcon(userEntry.change)}
                    <span className="text-sm font-semibold">{Math.abs(userEntry.change)}</span>
                  </div>
                </div>
              )
            })()}
          </Card>
        )}
      </div>

      <BottomNav />
    </div>
  )
}
