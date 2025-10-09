"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { MapPin, Clock, Users, ChevronRight } from "lucide-react"
import { format, formatDistanceToNow } from "date-fns"
import type { Activity } from "@/lib/types"
import { cn } from "@/lib/utils"

interface ActivityHistoryProps {
  activities: Activity[]
}

const activityTypeLabels = {
  walk: "Walk",
  playdate: "Playdate",
  training: "Training",
  other: "Other",
}

const activityTypeColors = {
  walk: "bg-primary/10 text-primary border-primary/20",
  playdate: "bg-secondary/10 text-secondary border-secondary/20",
  training: "bg-accent/10 text-accent border-accent/20",
  other: "bg-muted text-muted-foreground border-border",
}

export function ActivityHistory({ activities }: ActivityHistoryProps) {
  if (activities.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
          <MapPin className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">No activities yet</h3>
        <p className="text-sm text-muted-foreground">Start tracking your pet's activities!</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold">Recent Activity</h2>

      <div className="space-y-3">
        {activities.map((activity) => (
          <Card key={activity.id} className="glass p-4 hover:border-primary/50 transition-colors cursor-pointer">
            <div className="flex items-start gap-3">
              <div className="flex-1 space-y-3">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <Badge className={cn("border mb-2", activityTypeColors[activity.type])}>
                      {activityTypeLabels[activity.type]}
                    </Badge>
                    <p className="text-sm text-muted-foreground">
                      {formatDistanceToNow(new Date(activity.startTime), { addSuffix: true })}
                    </p>
                  </div>
                  <ChevronRight className="w-5 h-5 text-muted-foreground shrink-0" />
                </div>

                <div className="space-y-2">
                  {activity.distance && (
                    <div className="flex items-center gap-2 text-sm">
                      <MapPin className="w-4 h-4 text-muted-foreground" />
                      <span>{activity.distance.toFixed(1)} miles</span>
                    </div>
                  )}

                  <div className="flex items-center gap-2 text-sm">
                    <Clock className="w-4 h-4 text-muted-foreground" />
                    <span>
                      {Math.floor(activity.duration / 60)} min
                      {activity.endTime && ` • ${format(new Date(activity.startTime), "h:mm a")}`}
                    </span>
                  </div>

                  {activity.participants && activity.participants.length > 0 && (
                    <div className="flex items-center gap-2 text-sm">
                      <Users className="w-4 h-4 text-muted-foreground" />
                      <span>{activity.participants.length} participants</span>
                    </div>
                  )}
                </div>

                {activity.notes && <p className="text-sm text-muted-foreground line-clamp-2">{activity.notes}</p>}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
