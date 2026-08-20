"use client"

import { useState } from "react"
import { Loader2, RefreshCw, SlidersHorizontal, Sparkles } from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import { BottomNav } from "@/components/bottom-nav"
import { DiscoverMapView } from "@/components/discover/discover-map-view"
import { FilterSheet } from "@/components/discover/filter-sheet"
import { MatchCard } from "@/components/discover/match-card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { compatibilityApi } from "@/lib/api"
import { useSessionStore } from "@/store/session"

export default function DiscoverPage() {
  const user = useSessionStore((state) => state.user)
  const [filterOpen, setFilterOpen] = useState(false)
  const [activeTab, setActiveTab] = useState("matches")
  const primaryPetId = user?.pets?.[0]?.id

  const {
    data: matches = [],
    isLoading,
    isFetching,
    error,
    refetch,
  } = useQuery({
    queryKey: ["recommendations", primaryPetId],
    queryFn: () => compatibilityApi.getRecommendations(primaryPetId!),
    enabled: Boolean(primaryPetId),
    staleTime: 60_000,
  })

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-xl items-center justify-between gap-4 px-4 py-4">
          <div>
            <p className="eyebrow">Compatibility, not popularity</p>
            <h1 className="mt-1 text-2xl font-bold tracking-tight">Discover</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              {activeTab === "matches"
                ? primaryPetId
                  ? `${matches.length} profile-based ${matches.length === 1 ? "match" : "matches"}`
                  : "Add a pet profile to start matching"
                : "Explore nearby places and services"}
            </p>
          </div>
          <Button
            variant="outline"
            size="icon"
            aria-label="Open discovery filters"
            onClick={() => setFilterOpen(true)}
            className="shrink-0 bg-transparent"
          >
            <SlidersHorizontal className="h-5 w-5" aria-hidden="true" />
          </Button>
        </div>
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="sticky top-[89px] z-30 border-b border-border/50 bg-background/88 backdrop-blur-2xl">
          <TabsList className="mx-auto grid h-12 w-full max-w-xl grid-cols-2 bg-transparent px-4">
            <TabsTrigger value="matches" className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
              Compatible dogs
            </TabsTrigger>
            <TabsTrigger value="map" className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
              Map & services
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="matches" className="mt-0">
          <main id="main-content" className="mx-auto max-w-xl space-y-4 px-4 py-5">
            <section className="surface-soft flex items-start gap-3 rounded-2xl p-4" aria-labelledby="matching-explainer">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                <Sparkles className="h-5 w-5" aria-hidden="true" />
              </div>
              <div className="min-w-0 flex-1">
                <h2 id="matching-explainer" className="text-sm font-semibold">
                  A recommendation should explain itself
                </h2>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  Woof ranks known candidate relationships with a deterministic baseline built from pet profile signals.
                  Each result exposes confidence and the factors behind the score, while advanced models remain gated behind evaluation.
                </p>
              </div>
            </section>

            {isLoading ? (
              <div className="flex min-h-72 flex-col items-center justify-center gap-3" role="status">
                <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden="true" />
                <p className="text-sm text-muted-foreground">Ranking compatible dogs…</p>
              </div>
            ) : !primaryPetId ? (
              <div className="surface-soft flex min-h-72 flex-col items-center justify-center rounded-2xl px-6 text-center">
                <h2 className="text-lg font-semibold">Start with your dog</h2>
                <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                  Matching needs a pet profile so Woof can compare temperament, life stage and other available context.
                </p>
                <Button className="mt-5" onClick={() => setFilterOpen(true)}>
                  Review profile setup
                </Button>
              </div>
            ) : error ? (
              <div className="surface-soft flex min-h-72 flex-col items-center justify-center rounded-2xl px-6 text-center">
                <h2 className="text-lg font-semibold">Recommendations are temporarily unavailable</h2>
                <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                  Discovery failed without blocking the rest of Woof. Retry the ranking request when you are ready.
                </p>
                <Button variant="outline" className="mt-5 gap-2 bg-transparent" onClick={() => refetch()} disabled={isFetching}>
                  {isFetching ? (
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  ) : (
                    <RefreshCw className="h-4 w-4" aria-hidden="true" />
                  )}
                  Retry
                </Button>
              </div>
            ) : matches.length === 0 ? (
              <div className="surface-soft flex min-h-72 flex-col items-center justify-center rounded-2xl px-6 text-center">
                <h2 className="text-lg font-semibold">No candidate relationships yet</h2>
                <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                  Woof only ranks relationships the backend currently knows about. As the local network grows, compatible candidates will appear here.
                </p>
                <Button variant="outline" className="mt-5 bg-transparent" onClick={() => setFilterOpen(true)}>
                  Adjust discovery preferences
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {matches.map((match) => (
                  <MatchCard key={match.id} match={match} />
                ))}
                <div className="py-5 text-center">
                  <p className="text-sm text-muted-foreground">
                    You have reached the end of the current candidate set.
                  </p>
                  <Button variant="ghost" className="mt-2 gap-2" onClick={() => refetch()} disabled={isFetching}>
                    {isFetching ? (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    ) : (
                      <RefreshCw className="h-4 w-4" aria-hidden="true" />
                    )}
                    Refresh matches
                  </Button>
                </div>
              </div>
            )}
          </main>
        </TabsContent>

        <TabsContent value="map" className="mt-0">
          <DiscoverMapView />
        </TabsContent>
      </Tabs>

      <FilterSheet open={filterOpen} onOpenChange={setFilterOpen} />
      <BottomNav />
    </div>
  )
}
