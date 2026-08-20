"use client"

import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Calendar, MapPin, Clock, Users, Share2, CheckCircle2 } from "lucide-react"
import { format } from "date-fns"
import type { Event } from "@/lib/types"
import { cn } from "@/lib/utils"
import { CheckInSheet } from "./check-in-sheet"

interface EventDetailSheetProps {
  event: Event
  open: boolean
  onOpenChange: (open: boolean) => void
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

export function EventDetailSheet({ event, open, onOpenChange }: EventDetailSheetProps) {
  const [isAttending, setIsAttending] = useState(false)
  const [checkInOpen, setCheckInOpen] = useState(false)
  const spotsLeft = event.capacity - event.attendees.length
  const isFull = spotsLeft === 0

  const handleRSVP = () => {
    setIsAttending(!isAttending)
  }

  return (
    <>
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent side="bottom" className="h-[90vh] overflow-y-auto">
          <SheetHeader>
            <SheetTitle>Event Details</SheetTitle>
          </SheetHeader>

          <div className="space-y-6 py-6">
            {/* Event Image */}
            {event.imageUrl && (
              <div className="relative aspect-[16/9] rounded-lg overflow-hidden -mx-6">
                <img
                  src={event.imageUrl || "/placeholder.svg"}
                  alt={event.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-3 left-3">
                  <Badge className={cn("border", categoryColors[event.category])}>
                    {categoryLabels[event.category]}
                  </Badge>
                </div>
              </div>
            )}

            {/* Title & Description */}
            <div>
              <h2 className="text-2xl font-bold mb-3 text-balance">{event.title}</h2>
              <p className="text-muted-foreground">{event.description}</p>
            </div>

            {/* Event Info */}
            <div className="glass rounded-lg p-4 space-y-3">
              <div className="flex items-start gap-3">
                <Calendar className="w-5 h-5 text-primary mt-0.5 shrink-0" />
                <div>
                  <p className="font-medium">{format(new Date(event.datetime), "EEEE, MMMM d, yyyy")}</p>
                  <p className="text-sm text-muted-foreground">{format(new Date(event.datetime), "h:mm a")}</p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <Clock className="w-5 h-5 text-primary mt-0.5 shrink-0" />
                <div>
                  <p className="font-medium">{event.duration} minutes</p>
                  <p className="text-sm text-muted-foreground">Duration</p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <MapPin className="w-5 h-5 text-primary mt-0.5 shrink-0" />
                <div>
                  <p className="font-medium">{event.location.address}</p>
                  <Button variant="link" className="h-auto p-0 text-sm text-primary">
                    Get Directions
                  </Button>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <Users className="w-5 h-5 text-primary mt-0.5 shrink-0" />
                <div>
                  <p className="font-medium">
                    {event.attendees.length} / {event.capacity} attending
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {spotsLeft > 0 ? `${spotsLeft} spots left` : "Event is full"}
                  </p>
                </div>
              </div>
            </div>

            {/* Attendees */}
            {event.attendees.length > 0 && (
              <div className="space-y-3">
                <h3 className="font-semibold">Attendees ({event.attendees.length})</h3>
                <div className="flex flex-wrap gap-2">
                  {event.attendees.map((attendeeId, index) => (
                    <Avatar key={attendeeId} className="w-12 h-12 border-2 border-border">
                      <AvatarFallback className="bg-primary/10">{String.fromCharCode(65 + index)}</AvatarFallback>
                    </Avatar>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="space-y-3 pt-4">
              {isAttending ? (
                <>
                  <Button size="lg" className="w-full gap-2" onClick={() => setCheckInOpen(true)}>
                    <CheckCircle2 className="w-5 h-5" />
                    Check In
                  </Button>
                  <Button size="lg" variant="outline" className="w-full bg-transparent" onClick={handleRSVP}>
                    Cancel RSVP
                  </Button>
                </>
              ) : (
                <Button size="lg" className="w-full" onClick={handleRSVP} disabled={isFull}>
                  {isFull ? "Event Full" : "RSVP to Event"}
                </Button>
              )}

              <Button size="lg" variant="outline" className="w-full gap-2 bg-transparent">
                <Share2 className="w-5 h-5" />
                Share Event
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <CheckInSheet event={event} open={checkInOpen} onOpenChange={setCheckInOpen} />
    </>
  )
}
