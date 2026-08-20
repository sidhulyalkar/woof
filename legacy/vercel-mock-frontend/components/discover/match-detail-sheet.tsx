"use client"

import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { MapPin, Calendar, Activity, Heart, Users } from "lucide-react"
import type { Match } from "@/lib/types"
import { Button } from "@/components/ui/button"
import Link from "next/link"

interface MatchDetailSheetProps {
  match: Match
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function MatchDetailSheet({ match, open, onOpenChange }: MatchDetailSheetProps) {
  const factorIcons = {
    schedule: Calendar,
    location: MapPin,
    activityLevel: Activity,
    petCompatibility: Heart,
    interests: Users,
  }

  const factorLabels = {
    schedule: "Schedule Match",
    location: "Location",
    activityLevel: "Activity Level",
    petCompatibility: "Pet Compatibility",
    interests: "Shared Interests",
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="bottom" className="h-[90vh] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Match Details</SheetTitle>
        </SheetHeader>

        <div className="space-y-6 py-6">
          {/* Pet Image */}
          <div className="relative aspect-[4/3] rounded-lg overflow-hidden">
            <img
              src={match.pet.photoUrl || "/placeholder.svg"}
              alt={match.pet.name}
              className="w-full h-full object-cover"
            />
          </div>

          {/* Owner & Pet Info */}
          <div className="flex items-start gap-4">
            <Avatar className="w-16 h-16 border-2 border-border">
              <AvatarImage src={match.owner.avatarUrl || "/placeholder.svg"} />
              <AvatarFallback>{match.owner.name[0]}</AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <h2 className="text-2xl font-bold">
                {match.pet.name} & {match.owner.name}
              </h2>
              <p className="text-muted-foreground">
                {match.pet.breed} • {match.pet.age} years old
              </p>
              <div className="flex items-center gap-1 text-sm text-muted-foreground mt-1">
                <MapPin className="w-4 h-4" />
                <span>{match.distance} miles away</span>
              </div>
            </div>
          </div>

          {/* Overall Compatibility */}
          <div className="glass rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Overall Compatibility</h3>
              <span className="text-2xl font-bold text-primary">{match.compatibility.overall}%</span>
            </div>
            <Progress value={match.compatibility.overall} className="h-3" />
          </div>

          {/* Compatibility Factors */}
          <div className="space-y-3">
            <h3 className="font-semibold">Compatibility Breakdown</h3>
            {Object.entries(match.compatibility.factors).map(([key, value]) => {
              const Icon = factorIcons[key as keyof typeof factorIcons]
              const label = factorLabels[key as keyof typeof factorLabels]
              return (
                <div key={key} className="glass rounded-lg p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Icon className="w-4 h-4 text-primary" />
                      <span className="text-sm font-medium">{label}</span>
                    </div>
                    <span className="text-sm font-bold">{value}%</span>
                  </div>
                  <Progress value={value} className="h-2" />
                </div>
              )
            })}
          </div>

          {/* Why This Match */}
          <div className="space-y-3">
            <h3 className="font-semibold">Why This Match?</h3>
            <div className="space-y-2">
              {match.compatibility.explanation.map((reason, index) => (
                <div key={index} className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 shrink-0" />
                  <p className="text-sm text-muted-foreground">{reason}</p>
                </div>
              ))}
            </div>
          </div>

          {/* About */}
          <div className="space-y-3">
            <h3 className="font-semibold">About {match.owner.name}</h3>
            <p className="text-sm text-muted-foreground">{match.owner.bio}</p>
          </div>

          {/* Pet Details */}
          <div className="space-y-3">
            <h3 className="font-semibold">About {match.pet.name}</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="glass rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Species</p>
                <p className="font-medium capitalize">{match.pet.species}</p>
              </div>
              <div className="glass rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Size</p>
                <p className="font-medium capitalize">{match.pet.size}</p>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Temperament</p>
              <div className="flex flex-wrap gap-2">
                {match.pet.temperament.map((trait) => (
                  <Badge key={trait} variant="outline">
                    {trait}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

          {/* Action Button */}
          <Button asChild size="lg" className="w-full">
            <Link href={`/inbox?match=${match.id}`}>Send Message</Link>
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  )
}
