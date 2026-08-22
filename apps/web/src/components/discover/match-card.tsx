"use client"

import Link from "next/link"
import { useState } from "react"
import { CheckCircle2, Heart, Info, MessageCircle, ShieldCheck } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { AppImage } from "@/components/ui/app-image"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import type { Match } from "@/lib/types"
import { cn } from "@/lib/utils"
import { MatchDetailSheet } from "./match-detail-sheet"

interface MatchCardProps {
  match: Match
}

export function MatchCard({ match }: MatchCardProps) {
  const [liked, setLiked] = useState(false)
  const [detailOpen, setDetailOpen] = useState(false)

  const getScoreColor = (score: number) => {
    if (score >= 85) return "text-secondary"
    if (score >= 70) return "text-primary"
    return "text-muted-foreground"
  }

  const petDetails = [match.pet.breed, match.pet.age !== undefined ? `${match.pet.age} years` : undefined]
    .filter(Boolean)
    .join(" · ")

  return (
    <>
      <Card className="glass overflow-hidden rounded-2xl border-border/60">
        <div className="relative aspect-[4/3] overflow-hidden bg-muted/30">
          <AppImage
            src={match.pet.photoUrl || "/placeholder.svg"}
            alt={`${match.pet.name}, ${match.pet.breed || match.pet.species}`}
            width={1200}
            height={900}
            className="h-full w-full object-cover"
          />
          <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-background/80 to-transparent" aria-hidden="true" />

          <div className="absolute right-3 top-3">
            <div className="glass-strong flex items-center gap-2 rounded-full px-3 py-1.5">
              <span className={cn("h-2 w-2 rounded-full bg-current", getScoreColor(match.compatibility.overall))} />
              <span className={cn("text-sm font-bold", getScoreColor(match.compatibility.overall))}>
                {match.compatibility.overall}% match
              </span>
            </div>
          </div>

          <div className="absolute bottom-3 left-3 flex items-center gap-2">
            <Badge className="border-white/10 bg-background/80 text-foreground backdrop-blur-xl hover:bg-background/80">
              {match.compatibility.confidence}% confidence
            </Badge>
            {match.status === "CONFIRMED" && (
              <Badge className="border-secondary/20 bg-secondary/15 text-secondary hover:bg-secondary/15">
                <CheckCircle2 className="mr-1 h-3.5 w-3.5" aria-hidden="true" />
                Connected
              </Badge>
            )}
          </div>
        </div>

        <div className="space-y-4 p-4">
          <div className="flex items-start gap-3">
            <Avatar className="h-12 w-12 border-2 border-border">
              <AvatarImage src={match.owner.avatarUrl || "/placeholder.svg"} alt="" />
              <AvatarFallback>{match.owner.name.slice(0, 1).toUpperCase()}</AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-lg font-semibold tracking-tight">
                  {match.pet.name} <span className="font-normal text-muted-foreground">with</span> {match.owner.name}
                </h3>
                {match.owner.isVerified && (
                  <span className="inline-flex items-center text-secondary" title="Verified member">
                    <ShieldCheck className="h-4 w-4" aria-hidden="true" />
                    <span className="sr-only">Verified member</span>
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-sm capitalize text-muted-foreground">
                {petDetails || match.pet.species}
              </p>
            </div>
          </div>

          {match.owner.bio && (
            <p className="line-clamp-2 text-sm leading-relaxed text-muted-foreground">{match.owner.bio}</p>
          )}

          {match.compatibility.explanation.length > 0 && (
            <div className="flex flex-wrap gap-2" aria-label="Why this match">
              {match.compatibility.explanation.slice(0, 3).map((reason) => (
                <Badge key={reason} variant="secondary" className="bg-secondary/10 text-secondary hover:bg-secondary/15">
                  {reason}
                </Badge>
              ))}
            </div>
          )}

          {match.pet.temperament.length > 0 && (
            <div className="flex flex-wrap gap-2" aria-label={`${match.pet.name}'s temperament`}>
              {match.pet.temperament.slice(0, 4).map((trait) => (
                <Badge key={trait} variant="outline" className="text-xs text-muted-foreground">
                  {trait}
                </Badge>
              ))}
            </div>
          )}

          <div className="flex gap-2 border-t border-border/50 pt-4">
            <Button
              variant="outline"
              size="icon"
              aria-label={liked ? "Remove from favorites" : "Save match"}
              aria-pressed={liked}
              className={cn("bg-transparent", liked && "border-accent/60 text-accent")}
              onClick={() => setLiked(!liked)}
            >
              <Heart className={cn("h-5 w-5", liked && "fill-current")} aria-hidden="true" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              aria-label="View compatibility details"
              className="bg-transparent"
              onClick={() => setDetailOpen(true)}
            >
              <Info className="h-5 w-5" aria-hidden="true" />
            </Button>
            <Button asChild className="flex-1 gap-2">
              <Link href={`/inbox?match=${match.id}`}>
                <MessageCircle className="h-4 w-4" aria-hidden="true" />
                Start a conversation
              </Link>
            </Button>
          </div>
        </div>
      </Card>

      <MatchDetailSheet match={match} open={detailOpen} onOpenChange={setDetailOpen} />
    </>
  )
}
