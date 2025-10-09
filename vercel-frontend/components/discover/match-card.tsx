"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { MessageCircle, MapPin, Heart, Info } from "lucide-react"
import type { Match } from "@/lib/types"
import { cn } from "@/lib/utils"
import Link from "next/link"
import { MatchDetailSheet } from "./match-detail-sheet"

interface MatchCardProps {
  match: Match
}

export function MatchCard({ match }: MatchCardProps) {
  const [liked, setLiked] = useState(false)
  const [detailOpen, setDetailOpen] = useState(false)

  const getScoreColor = (score: number) => {
    if (score >= 90) return "text-primary"
    if (score >= 80) return "text-secondary"
    return "text-accent"
  }

  return (
    <>
      <Card className="glass overflow-hidden">
        {/* Pet Image */}
        <div className="relative aspect-[4/3] overflow-hidden">
          <img
            src={match.pet.photoUrl || "/placeholder.svg"}
            alt={match.pet.name}
            className="w-full h-full object-cover"
          />
          {/* Compatibility Score Badge */}
          <div className="absolute top-3 right-3">
            <div className="glass-strong rounded-full px-3 py-1.5 flex items-center gap-1.5">
              <div className={cn("w-2 h-2 rounded-full bg-current", getScoreColor(match.compatibility.overall))} />
              <span className={cn("text-sm font-bold", getScoreColor(match.compatibility.overall))}>
                {match.compatibility.overall}% Match
              </span>
            </div>
          </div>
          {/* Distance Badge */}
          <div className="absolute bottom-3 left-3">
            <div className="glass-strong rounded-full px-3 py-1.5 flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-muted-foreground" />
              <span className="text-sm font-medium">{match.distance} mi away</span>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Owner & Pet Info */}
          <div className="flex items-start gap-3">
            <Avatar className="w-12 h-12 border-2 border-border">
              <AvatarImage src={match.owner.avatarUrl || "/placeholder.svg"} />
              <AvatarFallback>{match.owner.name[0]}</AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-lg">
                {match.pet.name} & {match.owner.name}
              </h3>
              <p className="text-sm text-muted-foreground">
                {match.pet.breed} • {match.pet.age} years • {match.owner.age}
              </p>
            </div>
          </div>

          {/* Bio */}
          {match.owner.bio && <p className="text-sm text-muted-foreground line-clamp-2">{match.owner.bio}</p>}

          {/* Explainability Chips */}
          <div className="flex flex-wrap gap-2">
            {match.compatibility.explanation.slice(0, 3).map((reason, index) => (
              <Badge key={index} variant="secondary" className="text-xs bg-secondary/20 hover:bg-secondary/30">
                {reason}
              </Badge>
            ))}
          </div>

          {/* Temperament Tags */}
          <div className="flex flex-wrap gap-2">
            {match.pet.temperament.slice(0, 4).map((trait) => (
              <Badge key={trait} variant="outline" className="text-xs">
                {trait}
              </Badge>
            ))}
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            <Button
              variant="outline"
              size="icon"
              className={cn("bg-transparent", liked && "text-accent border-accent")}
              onClick={() => setLiked(!liked)}
            >
              <Heart className={cn("w-5 h-5", liked && "fill-current")} />
            </Button>
            <Button variant="outline" size="icon" className="bg-transparent" onClick={() => setDetailOpen(true)}>
              <Info className="w-5 h-5" />
            </Button>
            <Button asChild className="flex-1 gap-2">
              <Link href={`/inbox?match=${match.id}`}>
                <MessageCircle className="w-4 h-4" />
                Send Message
              </Link>
            </Button>
          </div>
        </div>
      </Card>

      <MatchDetailSheet match={match} open={detailOpen} onOpenChange={setDetailOpen} />
    </>
  )
}
