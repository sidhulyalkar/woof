"use client"

import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TrendingUp, MapPin, Clock, Footprints, Target, Flame, Zap } from "lucide-react"
import {
  Line,
  LineChart,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
  Pie,
  PieChart,
  Cell,
} from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

interface ActivityStatsProps {
  totalDistance: number
  totalDuration: number
  totalWalks: number
}

// Mock data for charts
const weeklyData = [
  { day: "Mon", distance: 2.1, duration: 45, calories: 180 },
  { day: "Tue", distance: 1.8, duration: 38, calories: 152 },
  { day: "Wed", distance: 2.5, duration: 52, calories: 210 },
  { day: "Thu", distance: 1.5, duration: 32, calories: 128 },
  { day: "Fri", distance: 3.2, duration: 68, calories: 272 },
  { day: "Sat", distance: 4.1, duration: 85, calories: 340 },
  { day: "Sun", distance: 2.8, duration: 58, calories: 232 },
]

const monthlyComparison = [
  { month: "Jan", distance: 45, walks: 18 },
  { month: "Feb", distance: 52, walks: 21 },
  { month: "Mar", distance: 48, walks: 19 },
  { month: "Apr", distance: 58, walks: 24 },
]

const activityBreakdown = [
  { name: "Walks", value: 65, color: "hsl(var(--chart-1))" },
  { name: "Runs", value: 20, color: "hsl(var(--chart-2))" },
  { name: "Playdates", value: 10, color: "hsl(var(--chart-3))" },
  { name: "Training", value: 5, color: "hsl(var(--chart-4))" },
]

export function ActivityStats({ totalDistance, totalDuration, totalWalks }: ActivityStatsProps) {
  const hours = Math.floor(totalDuration / 3600)
  const minutes = Math.floor((totalDuration % 3600) / 60)

  // Calculate additional metrics
  const avgPace = totalDistance > 0 ? totalDuration / 60 / totalDistance : 0
  const estimatedCalories = Math.round(totalDistance * 85) // Rough estimate
  const weeklyGoal = 15 // miles
  const goalProgress = (totalDistance / weeklyGoal) * 100

  return (
    <div className="space-y-4">
      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-3">
        <Card className="glass p-4 text-center">
          <MapPin className="w-5 h-5 text-primary mx-auto mb-2" />
          <p className="text-2xl font-bold">{totalDistance.toFixed(1)}</p>
          <p className="text-xs text-muted-foreground">Miles</p>
        </Card>

        <Card className="glass p-4 text-center">
          <Clock className="w-5 h-5 text-secondary mx-auto mb-2" />
          <p className="text-2xl font-bold">{hours > 0 ? `${hours}h` : `${minutes}m`}</p>
          <p className="text-xs text-muted-foreground">Time</p>
        </Card>

        <Card className="glass p-4 text-center">
          <Footprints className="w-5 h-5 text-accent mx-auto mb-2" />
          <p className="text-2xl font-bold">{totalWalks}</p>
          <p className="text-xs text-muted-foreground">Walks</p>
        </Card>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-3 gap-3">
        <Card className="glass p-3 text-center">
          <Zap className="w-4 h-4 text-yellow-500 mx-auto mb-1" />
          <p className="text-lg font-bold">{avgPace.toFixed(1)}</p>
          <p className="text-xs text-muted-foreground">Min/Mile</p>
        </Card>

        <Card className="glass p-3 text-center">
          <Flame className="w-4 h-4 text-orange-500 mx-auto mb-1" />
          <p className="text-lg font-bold">{estimatedCalories}</p>
          <p className="text-xs text-muted-foreground">Calories</p>
        </Card>

        <Card className="glass p-3 text-center">
          <Target className="w-4 h-4 text-green-500 mx-auto mb-1" />
          <p className="text-lg font-bold">{Math.min(goalProgress, 100).toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground">Goal</p>
        </Card>
      </div>

      {/* Weekly Goal Progress */}
      <Card className="glass p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            <h3 className="font-semibold text-sm">Weekly Goal</h3>
          </div>
          <span className="text-sm text-muted-foreground">
            {totalDistance.toFixed(1)} / {weeklyGoal} mi
          </span>
        </div>
        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-500"
            style={{ width: `${Math.min(goalProgress, 100)}%` }}
          />
        </div>
      </Card>

      {/* Charts Tabs */}
      <Tabs defaultValue="weekly" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-muted/50">
          <TabsTrigger value="weekly">Weekly</TabsTrigger>
          <TabsTrigger value="monthly">Monthly</TabsTrigger>
          <TabsTrigger value="breakdown">Breakdown</TabsTrigger>
        </TabsList>

        {/* Weekly Chart */}
        <TabsContent value="weekly" className="mt-4">
          <Card className="glass p-4">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-primary" />
              <h3 className="font-semibold">This Week's Activity</h3>
            </div>
            <ChartContainer
              config={{
                distance: {
                  label: "Distance (mi)",
                  color: "hsl(var(--chart-1))",
                },
                duration: {
                  label: "Duration (min)",
                  color: "hsl(var(--chart-2))",
                },
              }}
              className="h-[200px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={weeklyData}>
                  <defs>
                    <linearGradient id="colorDistance" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="day" fontSize={12} />
                  <YAxis fontSize={12} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Area
                    type="monotone"
                    dataKey="distance"
                    stroke="hsl(var(--chart-1))"
                    fill="url(#colorDistance)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </ChartContainer>
          </Card>

          {/* Calories Chart */}
          <Card className="glass p-4 mt-3">
            <div className="flex items-center gap-2 mb-4">
              <Flame className="w-5 h-5 text-orange-500" />
              <h3 className="font-semibold">Calories Burned</h3>
            </div>
            <ChartContainer
              config={{
                calories: {
                  label: "Calories",
                  color: "hsl(var(--chart-3))",
                },
              }}
              className="h-[180px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weeklyData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="day" fontSize={12} />
                  <YAxis fontSize={12} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="calories" fill="hsl(var(--chart-3))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </Card>
        </TabsContent>

        {/* Monthly Comparison */}
        <TabsContent value="monthly" className="mt-4">
          <Card className="glass p-4">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-primary" />
              <h3 className="font-semibold">Monthly Trends</h3>
            </div>
            <ChartContainer
              config={{
                distance: {
                  label: "Distance (mi)",
                  color: "hsl(var(--chart-1))",
                },
                walks: {
                  label: "Walks",
                  color: "hsl(var(--chart-2))",
                },
              }}
              className="h-[250px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={monthlyComparison}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="month" fontSize={12} />
                  <YAxis fontSize={12} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="distance"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                  <Line type="monotone" dataKey="walks" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </Card>

          {/* Stats comparison */}
          <div className="grid grid-cols-2 gap-3 mt-3">
            <Card className="glass p-4">
              <p className="text-xs text-muted-foreground mb-1">Avg Distance/Month</p>
              <p className="text-2xl font-bold">50.8 mi</p>
              <p className="text-xs text-green-500 mt-1">↑ 12% from last month</p>
            </Card>
            <Card className="glass p-4">
              <p className="text-xs text-muted-foreground mb-1">Avg Walks/Month</p>
              <p className="text-2xl font-bold">20.5</p>
              <p className="text-xs text-green-500 mt-1">↑ 8% from last month</p>
            </Card>
          </div>
        </TabsContent>

        {/* Activity Breakdown */}
        <TabsContent value="breakdown" className="mt-4">
          <Card className="glass p-4">
            <div className="flex items-center gap-2 mb-4">
              <Footprints className="w-5 h-5 text-primary" />
              <h3 className="font-semibold">Activity Types</h3>
            </div>
            <div className="flex items-center justify-center">
              <ChartContainer
                config={{
                  walks: { label: "Walks", color: "hsl(var(--chart-1))" },
                  runs: { label: "Runs", color: "hsl(var(--chart-2))" },
                  playdates: { label: "Playdates", color: "hsl(var(--chart-3))" },
                  training: { label: "Training", color: "hsl(var(--chart-4))" },
                }}
                className="h-[200px] w-full"
              >
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={activityBreakdown}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {activityBreakdown.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <ChartTooltip content={<ChartTooltipContent />} />
                  </PieChart>
                </ResponsiveContainer>
              </ChartContainer>
            </div>

            {/* Legend */}
            <div className="grid grid-cols-2 gap-2 mt-4">
              {activityBreakdown.map((item) => (
                <div key={item.name} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-sm">{item.name}</span>
                  <span className="text-sm text-muted-foreground ml-auto">{item.value}%</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Activity Stats */}
          <div className="grid grid-cols-2 gap-3 mt-3">
            <Card className="glass p-4">
              <p className="text-xs text-muted-foreground mb-1">Most Active Day</p>
              <p className="text-xl font-bold">Saturday</p>
              <p className="text-xs text-muted-foreground mt-1">4.1 mi average</p>
            </Card>
            <Card className="glass p-4">
              <p className="text-xs text-muted-foreground mb-1">Favorite Time</p>
              <p className="text-xl font-bold">Morning</p>
              <p className="text-xs text-muted-foreground mt-1">7-9 AM</p>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
