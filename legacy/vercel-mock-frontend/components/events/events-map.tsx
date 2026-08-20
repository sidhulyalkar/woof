"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { MapPin } from "lucide-react"
import type { Event } from "@/lib/types"
import { cn } from "@/lib/utils"

interface EventsMapProps {
  events: Event[]
  onSelectEvent: (event: Event) => void
}

const categoryColors = {
  playdate: "bg-primary text-primary-foreground",
  training: "bg-secondary text-secondary-foreground",
  social: "bg-accent text-accent-foreground",
  other: "bg-muted text-muted-foreground",
}

export function EventsMap({ events, onSelectEvent }: EventsMapProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  return (
    <div className="relative h-[calc(100vh-13rem)]">
      {/* Map Placeholder */}
      <div className="absolute inset-0 bg-gradient-to-br from-card via-background to-card">
        <div className="absolute inset-0 opacity-10">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
          </svg>
        </div>

        {/* Map Markers */}
        <div className="relative w-full h-full">
          {events.map((event, index) => {
            const top = 20 + ((index * 15) % 60)
            const left = 15 + ((index * 20) % 70)

            return (
              <button
                key={event.id}
                onClick={() => {
                  setSelectedId(event.id)
                  onSelectEvent(event)
                }}
                className={cn(
                  "absolute w-10 h-10 rounded-full flex items-center justify-center shadow-lg transition-transform hover:scale-110",
                  categoryColors[event.category],
                  selectedId === event.id && "scale-125 ring-4 ring-ring",
                )}
                style={{ top: `${top}%`, left: `${left}%` }}
              >
                <MapPin className="w-5 h-5" />
              </button>
            )
          })}
        </div>
      </div>

      {/* Legend */}
      <Card className="absolute bottom-4 left-4 right-4 glass p-4">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <span className="text-sm font-medium">Categories:</span>
          <div className="flex gap-2 flex-wrap">
            <Badge className={categoryColors.playdate}>Playdate</Badge>
            <Badge className={categoryColors.training}>Training</Badge>
            <Badge className={categoryColors.social}>Social</Badge>
            <Badge className={categoryColors.other}>Other</Badge>
          </div>
        </div>
      </Card>
    </div>
  )
}
