"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Calendar, MapPin, Clock, Users } from "lucide-react"
import { format } from "date-fns"
import type { Event } from "@/lib/types"
import { cn } from "@/lib/utils"

interface EventCardProps {
  event: Event
  onViewDetails: () => void
}

const categoryColors = {
  playdate: "bg-primary/10 text-primary border-primary/20",
  training: "bg-secondary/10 text-secondary border-secondary/20",
  social: "bg-accent/10 text-accent border-accent/20",
  other: "bg-muted text-muted-foreground border-border",
}

const categoryLabels = {
  playdate: "Playdate",
  training: "Training",
  social: "Social",
  other: "Other",
}

export function EventCard({ event, onViewDetails }: EventCardProps) {
  const spotsLeft = event.capacity - event.attendees.length
  const isAlmostFull = spotsLeft <= 3
  const isFull = spotsLeft === 0

  return (
    <Card className="glass overflow-hidden">
      {/* Event Image */}
      {event.imageUrl && (
        <div className="relative aspect-[16/9] overflow-hidden">
          <img src={event.imageUrl || "/placeholder.svg"} alt={event.title} className="w-full h-full object-cover" />
          <div className="absolute top-3 left-3">
            <Badge className={cn("border", categoryColors[event.category])}>{categoryLabels[event.category]}</Badge>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="p-4 space-y-4">
        <div>
          <h3 className="text-lg font-semibold mb-2 text-balance">{event.title}</h3>
          <p className="text-sm text-muted-foreground line-clamp-2">{event.description}</p>
        </div>

        {/* Event Details */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <Calendar className="w-4 h-4 text-muted-foreground shrink-0" />
            <span>{format(new Date(event.datetime), "EEE, MMM d 'at' h:mm a")}</span>
          </div>

          <div className="flex items-center gap-2 text-sm">
            <Clock className="w-4 h-4 text-muted-foreground shrink-0" />
            <span>{event.duration} minutes</span>
          </div>

          <div className="flex items-center gap-2 text-sm">
            <MapPin className="w-4 h-4 text-muted-foreground shrink-0" />
            <span className="line-clamp-1">{event.location.address}</span>
          </div>

          <div className="flex items-center gap-2 text-sm">
            <Users className="w-4 h-4 text-muted-foreground shrink-0" />
            <span>
              {event.attendees.length} / {event.capacity} attending
            </span>
            {isAlmostFull && !isFull && (
              <Badge variant="outline" className="text-xs bg-accent/10 text-accent border-accent/20">
                Almost Full
              </Badge>
            )}
            {isFull && (
              <Badge variant="outline" className="text-xs bg-destructive/10 text-destructive border-destructive/20">
                Full
              </Badge>
            )}
          </div>
        </div>

        {/* Attendees Preview */}
        {event.attendees.length > 0 && (
          <div className="flex items-center gap-2">
            <div className="flex -space-x-2">
              {event.attendees.slice(0, 4).map((attendeeId, index) => (
                <Avatar key={attendeeId} className="w-8 h-8 border-2 border-card">
                  <AvatarFallback className="text-xs bg-primary/10">{String.fromCharCode(65 + index)}</AvatarFallback>
                </Avatar>
              ))}
            </div>
            {event.attendees.length > 4 && (
              <span className="text-xs text-muted-foreground">+{event.attendees.length - 4} more</span>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-2 pt-2">
          <Button variant="outline" onClick={onViewDetails} className="flex-1 bg-transparent">
            View Details
          </Button>
          <Button className="flex-1" disabled={isFull}>
            {isFull ? "Event Full" : "RSVP"}
          </Button>
        </div>
      </div>
    </Card>
  )
}
