"use client"

import { useState } from "react"
import { BottomNav } from "@/components/bottom-nav"
import { MatchCard } from "@/components/discover/match-card"
import { FilterSheet } from "@/components/discover/filter-sheet"
import { DiscoverMapView } from "@/components/discover/discover-map-view"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { SlidersHorizontal, Sparkles, Loader2 } from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import { compatibilityApi } from "@/lib/api"
import { useSessionStore } from "@/store/session"
import type { Match } from "@/lib/types"

// Mock data for demonstration
const mockMatches: Match[] = [
  {
    id: "1",
    owner: {
      id: "o1",
      name: "Sarah Johnson",
      age: 32,
      location: { lat: 37.7749, lng: -122.4194, address: "San Francisco, CA" },
      bio: "Love hiking with my golden retriever! Always looking for new trails and dog-friendly spots.",
      avatarUrl: "/woman-and-loyal-companion.png",
      preferences: {
        activityLevel: "high",
        schedule: ["Weekday mornings", "Weekends"],
        interests: ["Hiking", "Dog parks", "Training"],
      },
    },
    pet: {
      id: "p1",
      ownerId: "o1",
      name: "Max",
      species: "dog",
      breed: "Golden Retriever",
      age: 3,
      size: "large",
      temperament: ["Friendly", "Energetic", "Social"],
      photoUrl: "/golden-retriever.png",
    },
    compatibility: {
      overall: 94,
      factors: {
        schedule: 95,
        location: 98,
        activityLevel: 92,
        petCompatibility: 90,
        interests: 95,
      },
      explanation: [
        "Similar activity schedules",
        "Lives nearby (0.8 mi)",
        "Both love hiking",
        "Pets have compatible energy",
      ],
    },
    distance: 0.8,
    matchedAt: new Date().toISOString(),
  },
  {
    id: "2",
    owner: {
      id: "o2",
      name: "Michael Chen",
      age: 28,
      location: { lat: 37.7849, lng: -122.4094, address: "San Francisco, CA" },
      bio: "Tech professional who loves taking my husky on weekend adventures. Looking for active pet parents!",
      avatarUrl: "/man-with-husky.jpg",
      preferences: {
        activityLevel: "high",
        schedule: ["Weekday evenings", "Weekends"],
        interests: ["Hiking", "Beach", "Dog parks"],
      },
    },
    pet: {
      id: "p2",
      ownerId: "o2",
      name: "Luna",
      species: "dog",
      breed: "Siberian Husky",
      age: 2,
      size: "large",
      temperament: ["Energetic", "Playful", "Social"],
      photoUrl: "/siberian-husky-portrait.png",
    },
    compatibility: {
      overall: 88,
      factors: {
        schedule: 85,
        location: 95,
        activityLevel: 90,
        petCompatibility: 88,
        interests: 82,
      },
      explanation: ["Compatible schedules", "Very close by (1.2 mi)", "High energy match", "Similar interests"],
    },
    distance: 1.2,
    matchedAt: new Date().toISOString(),
  },
  {
    id: "3",
    owner: {
      id: "o3",
      name: "Emily Rodriguez",
      age: 35,
      location: { lat: 37.7649, lng: -122.4294, address: "San Francisco, CA" },
      bio: "Yoga instructor and dog mom. Love peaceful morning walks and dog-friendly cafes.",
      avatarUrl: "/yoga-instructor.png",
      preferences: {
        activityLevel: "medium",
        schedule: ["Weekday mornings", "Weekday afternoons"],
        interests: ["Dog parks", "Pet cafes", "Training classes"],
      },
    },
    pet: {
      id: "p3",
      ownerId: "o3",
      name: "Bella",
      species: "dog",
      breed: "Australian Shepherd",
      age: 4,
      size: "medium",
      temperament: ["Friendly", "Calm", "Social"],
      photoUrl: "/australian-shepherd-portrait.png",
    },
    compatibility: {
      overall: 82,
      factors: {
        schedule: 88,
        location: 92,
        activityLevel: 75,
        petCompatibility: 85,
        interests: 70,
      },
      explanation: [
        "Morning availability match",
        "Close proximity (1.5 mi)",
        "Balanced activity levels",
        "Friendly pets",
      ],
    },
    distance: 1.5,
    matchedAt: new Date().toISOString(),
  },
]

export default function DiscoverPage() {
  const user = useSessionStore(state => state.user)
  const [filterOpen, setFilterOpen] = useState(false)
  const [activeTab, setActiveTab] = useState("matches")

  // Get primary pet ID - will use first pet for now
  const primaryPetId = user?.pets?.[0]?.id

  // Fetch recommendations
  const { data: matches = [], isLoading } = useQuery({
    queryKey: ['recommendations', primaryPetId],
    queryFn: () => {
      if (!primaryPetId) return Promise.resolve([])
      return compatibilityApi.getRecommendations(primaryPetId)
    },
    enabled: !!primaryPetId,
  })

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="flex items-center justify-between px-4 py-4 max-w-lg mx-auto">
          <div>
            <h1 className="text-2xl font-bold">Discover</h1>
            <p className="text-sm text-muted-foreground">
              {activeTab === "matches" ? `${matches.length} compatible matches` : "Nearby services & activities"}
            </p>
          </div>
          <Button variant="outline" size="icon" onClick={() => setFilterOpen(true)} className="bg-transparent">
            <SlidersHorizontal className="w-5 h-5" />
          </Button>
        </div>
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="sticky top-[73px] z-30 glass-strong border-b border-border/50">
          <TabsList className="w-full max-w-lg mx-auto grid grid-cols-2 bg-transparent h-12">
            <TabsTrigger value="matches" className="data-[state=active]:bg-primary/10">
              Pet Matches
            </TabsTrigger>
            <TabsTrigger value="map" className="data-[state=active]:bg-primary/10">
              Map & Services
            </TabsTrigger>
          </TabsList>
        </div>

        {/* Matches View */}
        <TabsContent value="matches" className="mt-0">
          <main className="px-4 py-6 max-w-lg mx-auto space-y-4">
            {/* Info Banner */}
            <div className="glass rounded-lg p-4 flex items-start gap-3">
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                <Sparkles className="w-5 h-5 text-primary" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-sm">Smart Matching</h3>
                <p className="text-xs text-muted-foreground mt-1">
                  Matches are ranked by compatibility based on your preferences, location, and pet personality.
                </p>
              </div>
            </div>

            {/* Match Cards */}
            <div className="space-y-4">
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
              ) : !primaryPetId ? (
                <div className="text-center py-12 px-4">
                  <p className="text-muted-foreground mb-2">No pet profile found</p>
                  <p className="text-sm text-muted-foreground">Please add a pet to see matches!</p>
                </div>
              ) : matches.length === 0 ? (
                <div className="text-center py-12 px-4">
                  <p className="text-muted-foreground mb-2">No matches found</p>
                  <p className="text-sm text-muted-foreground">Check back soon for new connections!</p>
                </div>
              ) : (
                matches.map((match) => (
                  <MatchCard key={match.id} match={match} />
                ))
              )}
            </div>

            {/* Load More */}
            {matches.length > 0 && (
              <div className="text-center py-8">
                <p className="text-sm text-muted-foreground">You've seen all matches in your area</p>
                <Button variant="outline" className="mt-4 bg-transparent" onClick={() => setFilterOpen(true)}>
                  Adjust Filters
                </Button>
              </div>
            )}
          </main>
        </TabsContent>

        {/* Map View */}
        <TabsContent value="map" className="mt-0">
          <DiscoverMapView />
        </TabsContent>
      </Tabs>

      <FilterSheet open={filterOpen} onOpenChange={setFilterOpen} />
      <BottomNav />
    </div>
  )
}
