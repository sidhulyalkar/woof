"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { MapPin, Calendar, Check, X } from "lucide-react"
import { format } from "date-fns"
import { cn } from "@/lib/utils"

interface MeetupProposalCardProps {
  location: string
  datetime: string
  isCurrentUser: boolean
}

export function MeetupProposalCard({ location, datetime, isCurrentUser }: MeetupProposalCardProps) {
  return (
    <Card className={cn("glass p-4 space-y-3 max-w-[85%]", isCurrentUser && "border-primary/50")}>
      <div className="flex items-center gap-2 text-primary">
        <Calendar className="w-4 h-4" />
        <span className="text-sm font-semibold">Meetup Proposal</span>
      </div>

      <div className="space-y-2">
        <div className="flex items-start gap-2">
          <MapPin className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium">{location}</p>
            <p className="text-xs text-muted-foreground">Location</p>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <Calendar className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium">{format(new Date(datetime), "MMM d, yyyy 'at' h:mm a")}</p>
            <p className="text-xs text-muted-foreground">Date & Time</p>
          </div>
        </div>
      </div>

      {!isCurrentUser && (
        <div className="flex gap-2 pt-2">
          <Button size="sm" className="flex-1 gap-2">
            <Check className="w-4 h-4" />
            Accept
          </Button>
          <Button size="sm" variant="outline" className="flex-1 gap-2 bg-transparent">
            <X className="w-4 h-4" />
            Decline
          </Button>
        </div>
      )}

      {isCurrentUser && <p className="text-xs text-muted-foreground text-center pt-2">Waiting for response...</p>}
    </Card>
  )
}
