"use client"

import Link from "next/link"
import { Calendar, Dna, Heart, PawPrint, ShieldCheck, Sparkles } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import type { Match } from "@/lib/types"

interface MatchDetailSheetProps {
  match: Match
  open: boolean
  onOpenChange: (open: boolean) => void
}

const factorMeta = {
  species: { icon: PawPrint, label: "Species fit" },
  temperament: { icon: Heart, label: "Temperament" },
  age: { icon: Calendar, label: "Life stage" },
  breed: { icon: Dna, label: "Breed context" },
}

export function MatchDetailSheet({ match, open, onOpenChange }: MatchDetailSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="bottom" className="h-[92vh] overflow-y-auto rounded-t-3xl border-border/70">
        <SheetHeader>
          <SheetTitle className="text-left">Why Woof suggested this match</SheetTitle>
        </SheetHeader>

        <div className="mx-auto max-w-xl space-y-6 py-6">
          <div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-muted/30">
            <img
              src={match.pet.photoUrl || "/placeholder.svg"}
              alt={`${match.pet.name}, ${match.pet.breed || match.pet.species}`}
              className="h-full w-full object-cover"
            />
            <div className="absolute inset-x-0 bottom-0 h-28 bg-gradient-to-t from-background/85 to-transparent" aria-hidden="true" />
            <div className="absolute bottom-4 left-4 flex flex-wrap gap-2">
              <Badge className="bg-background/85 text-foreground hover:bg-background/85">
                {match.compatibility.overall}% match
              </Badge>
              <Badge className="bg-background/85 text-muted-foreground hover:bg-background/85">
                {match.compatibility.confidence}% confidence
              </Badge>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <Avatar className="h-14 w-14 border-2 border-border">
              <AvatarImage src={match.owner.avatarUrl || "/placeholder.svg"} alt="" />
              <AvatarFallback>{match.owner.name.slice(0, 1).toUpperCase()}</AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-2xl font-bold tracking-tight">
                  {match.pet.name} <span className="font-normal text-muted-foreground">with</span> {match.owner.name}
                </h2>
                {match.owner.isVerified && (
                  <span className="inline-flex items-center gap-1 text-xs font-semibold text-secondary">
                    <ShieldCheck className="h-4 w-4" aria-hidden="true" />
                    Verified
                  </span>
                )}
              </div>
              <p className="mt-1 text-sm capitalize text-muted-foreground">
                {[match.pet.breed, match.pet.age !== undefined ? `${match.pet.age} years old` : undefined]
                  .filter(Boolean)
                  .join(" · ") || match.pet.species}
              </p>
            </div>
          </div>

          <section className="glass rounded-2xl p-4" aria-labelledby="compatibility-summary">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="eyebrow">Compatibility</p>
                <h3 id="compatibility-summary" className="mt-1 font-semibold">
                  Profile-based estimate
                </h3>
              </div>
              <span className="text-2xl font-bold text-secondary">{match.compatibility.overall}%</span>
            </div>
            <Progress value={match.compatibility.overall} className="mt-4 h-2.5" />
            <div className="mt-4 flex items-start gap-2 rounded-xl bg-muted/40 p-3 text-sm text-muted-foreground">
              <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
              <p>
                This score comes from <span className="font-medium text-foreground">{match.compatibility.source}</span>.
                It is a recommendation aid, not a guarantee of how two dogs will behave together.
              </p>
            </div>
          </section>

          <section className="space-y-3" aria-labelledby="factor-heading">
            <div>
              <p className="eyebrow">Signals</p>
              <h3 id="factor-heading" className="mt-1 font-semibold">
                Compatibility breakdown
              </h3>
            </div>
            {Object.entries(match.compatibility.factors).map(([key, value]) => {
              if (value === undefined) return null
              const meta = factorMeta[key as keyof typeof factorMeta]
              if (!meta) return null
              const Icon = meta.icon

              return (
                <div key={key} className="surface-soft rounded-2xl p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
                      <span className="text-sm font-medium">{meta.label}</span>
                    </div>
                    <span className="text-sm font-bold">{value}%</span>
                  </div>
                  <Progress value={value} className="mt-3 h-2" />
                </div>
              )
            })}
          </section>

          {match.compatibility.explanation.length > 0 && (
            <section className="space-y-3" aria-labelledby="reason-heading">
              <div>
                <p className="eyebrow">Explanation</p>
                <h3 id="reason-heading" className="mt-1 font-semibold">
                  Why this match surfaced
                </h3>
              </div>
              <div className="space-y-2">
                {match.compatibility.explanation.map((reason) => (
                  <div key={reason} className="flex items-start gap-3 rounded-xl bg-muted/25 p-3">
                    <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-secondary" aria-hidden="true" />
                    <p className="text-sm leading-relaxed text-muted-foreground">{reason}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {match.owner.bio && (
            <section className="space-y-2" aria-labelledby="owner-heading">
              <h3 id="owner-heading" className="font-semibold">
                About {match.owner.name}
              </h3>
              <p className="text-sm leading-relaxed text-muted-foreground">{match.owner.bio}</p>
            </section>
          )}

          {match.pet.temperament.length > 0 && (
            <section className="space-y-2" aria-labelledby="temperament-heading">
              <h3 id="temperament-heading" className="font-semibold">
                {match.pet.name}&apos;s temperament
              </h3>
              <div className="flex flex-wrap gap-2">
                {match.pet.temperament.map((trait) => (
                  <Badge key={trait} variant="outline" className="text-muted-foreground">
                    {trait}
                  </Badge>
                ))}
              </div>
            </section>
          )}

          <Button asChild size="lg" className="w-full">
            <Link href={`/inbox?match=${match.id}`}>Start a conversation</Link>
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  )
}
