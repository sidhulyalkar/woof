"use client"

import { useState } from "react"
import { BottomNav } from "@/components/bottom-nav"
import { EventCard } from "@/components/events/event-card"
import { EventDetailSheet } from "@/components/events/event-detail-sheet"
import { EventsMap } from "@/components/events/events-map"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Plus, MapIcon, List } from "lucide-react"
import type { Event } from "@/lib/types"

// Mock events data
const mockEvents: Event[] = [
  {
    id: "e1",
    title: "Weekend Dog Park Meetup",
    description:
      "Join us for a fun morning at the dog park! Great opportunity for your pups to socialize and make new friends. We'll have treats and water available.",
    location: {
      lat: 37.7749,
      lng: -122.4194,
      address: "Golden Gate Park, San Francisco, CA",
    },
    datetime: new Date(Date.now() + 1000 * 60 * 60 * 24 * 2).toISOString(),
    duration: 120,
    capacity: 20,
    attendees: ["o1", "o2", "o3", "o4", "o5"],
    organizerId: "o1",
    category: "playdate",
    imageUrl: "/dog-park-meetup.jpg",
  },
  {
    id: "e2",
    title: "Puppy Training Class",
    description:
      "Professional trainer-led session covering basic commands, leash training, and socialization. Perfect for puppies 3-12 months old.",
    location: {
      lat: 37.7849,
      lng: -122.4094,
      address: "Mission Bay Dog Training Center, San Francisco, CA",
    },
    datetime: new Date(Date.now() + 1000 * 60 * 60 * 24 * 5).toISOString(),
    duration: 90,
    capacity: 12,
    attendees: ["o2", "o6", "o7"],
    organizerId: "o2",
    category: "training",
    imageUrl: "/puppy-training.jpg",
  },
  {
    id: "e3",
    title: "Beach Day for Dogs",
    description:
      "Let's hit the beach! Bring your water-loving pups for a day of fun in the sun. Swimming, fetch, and beach games. Don't forget sunscreen!",
    location: {
      lat: 37.7649,
      lng: -122.5094,
      address: "Ocean Beach, San Francisco, CA",
    },
    datetime: new Date(Date.now() + 1000 * 60 * 60 * 24 * 7).toISOString(),
    duration: 180,
    capacity: 30,
    attendees: ["o1", "o3", "o8", "o9"],
    organizerId: "o3",
    category: "social",
    imageUrl: "/beach-dogs.jpg",
  },
  {
    id: "e4",
    title: "Pet Yoga & Wellness",
    description:
      "Relaxing yoga session with your furry companion. Learn stretches and exercises that benefit both you and your pet. All levels welcome!",
    location: {
      lat: 37.7549,
      lng: -122.4294,
      address: "Zen Pet Studio, San Francisco, CA",
    },
    datetime: new Date(Date.now() + 1000 * 60 * 60 * 24 * 3).toISOString(),
    duration: 60,
    capacity: 15,
    attendees: ["o4", "o5"],
    organizerId: "o4",
    category: "other",
    imageUrl: "/pet-yoga.jpg",
  },
]

export default function EventsPage() {
  const [events] = useState<Event[]>(mockEvents)
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null)
  const [view, setView] = useState<"list" | "map">("list")

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="px-4 py-4 max-w-lg mx-auto">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h1 className="text-2xl font-bold">Events</h1>
              <p className="text-sm text-muted-foreground">{events.length} upcoming events</p>
            </div>
            <Button size="icon" className="shrink-0">
              <Plus className="w-5 h-5" />
            </Button>
          </div>

          <Tabs value={view} onValueChange={(v) => setView(v as "list" | "map")} className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="list" className="gap-2">
                <List className="w-4 h-4" />
                List
              </TabsTrigger>
              <TabsTrigger value="map" className="gap-2">
                <MapIcon className="w-4 h-4" />
                Map
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-lg mx-auto">
        {view === "list" ? (
          <div className="px-4 py-6 space-y-4">
            {events.map((event) => (
              <EventCard key={event.id} event={event} onViewDetails={() => setSelectedEvent(event)} />
            ))}
          </div>
        ) : (
          <EventsMap events={events} onSelectEvent={setSelectedEvent} />
        )}
      </main>

      {selectedEvent && (
        <EventDetailSheet event={selectedEvent} open={!!selectedEvent} onOpenChange={() => setSelectedEvent(null)} />
      )}

      <BottomNav />
    </div>
  )
}
