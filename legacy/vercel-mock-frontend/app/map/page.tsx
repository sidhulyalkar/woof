"use client"

import { useState } from "react"
import { MapPin, Navigation, Users, Calendar, Briefcase } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { BottomNav } from "@/components/bottom-nav"
import type { MapMarker } from "@/lib/types"

export default function MapPage() {
  const [filter, setFilter] = useState<"all" | "pets" | "events" | "services">("all")
  const [selectedMarker, setSelectedMarker] = useState<MapMarker | null>(null)

  // Mock map markers
  const markers: MapMarker[] = [
    {
      id: "1",
      type: "pet",
      lat: 37.7749,
      lng: -122.4194,
      title: "Luna",
      subtitle: "with Sarah Chen",
      avatarUrl: "/border-collie.jpg",
      data: { distance: 0.3, compatibility: 92 },
    },
    {
      id: "2",
      type: "event",
      lat: 37.7699,
      lng: -122.4194,
      title: "Dog Park Meetup",
      subtitle: "Today at 3:00 PM",
      data: { attendees: 12, capacity: 20 },
    },
    {
      id: "3",
      type: "service",
      lat: 37.7799,
      lng: -122.4194,
      title: "Paws & Play Grooming",
      subtitle: "4.8 ★ • Pet Grooming",
      data: { rating: 4.8, reviews: 124 },
    },
  ]

  const filteredMarkers =
    filter === "all" ? markers : markers.filter((m) => m.type === (filter.replace("s", "") as any))

  const getMarkerIcon = (type: MapMarker["type"]) => {
    switch (type) {
      case "pet":
        return <Users className="h-4 w-4" />
      case "event":
        return <Calendar className="h-4 w-4" />
      case "service":
        return <Briefcase className="h-4 w-4" />
    }
  }

  return (
    <div className="relative h-screen">
      {/* Map Container */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-secondary/5">
        {/* Placeholder for actual map */}
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <MapPin className="mx-auto h-16 w-16 text-muted-foreground/30" />
            <p className="mt-4 text-sm text-muted-foreground">Map view would render here</p>
            <p className="text-xs text-muted-foreground">Showing {filteredMarkers.length} nearby locations</p>
          </div>
        </div>

        {/* Map Markers Visualization */}
        <div className="absolute inset-0 pointer-events-none">
          {filteredMarkers.map((marker, index) => (
            <div
              key={marker.id}
              className="absolute pointer-events-auto"
              style={{
                left: `${30 + index * 20}%`,
                top: `${40 + index * 10}%`,
              }}
              onClick={() => setSelectedMarker(marker)}
            >
              <div
                className={`flex h-10 w-10 items-center justify-center rounded-full border-2 border-background shadow-lg ${
                  marker.type === "pet" ? "bg-accent" : marker.type === "event" ? "bg-secondary" : "bg-primary"
                }`}
              >
                {marker.avatarUrl ? (
                  <Avatar className="h-8 w-8">
                    <AvatarImage src={marker.avatarUrl || "/placeholder.svg"} />
                    <AvatarFallback>{marker.title[0]}</AvatarFallback>
                  </Avatar>
                ) : (
                  <div className="text-white">{getMarkerIcon(marker.type)}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top Controls */}
      <div className="absolute top-0 left-0 right-0 z-10 p-4">
        <div className="flex items-center gap-2">
          <Button size="icon" variant="secondary" className="h-10 w-10 rounded-full shadow-lg">
            <Navigation className="h-5 w-5" />
          </Button>
          <div className="flex flex-1 gap-2 overflow-x-auto">
            <Button
              size="sm"
              variant={filter === "all" ? "default" : "secondary"}
              onClick={() => setFilter("all")}
              className="rounded-full shadow-lg"
            >
              All
            </Button>
            <Button
              size="sm"
              variant={filter === "pets" ? "default" : "secondary"}
              onClick={() => setFilter("pets")}
              className="rounded-full shadow-lg"
            >
              <Users className="mr-1 h-4 w-4" />
              Pets
            </Button>
            <Button
              size="sm"
              variant={filter === "events" ? "default" : "secondary"}
              onClick={() => setFilter("events")}
              className="rounded-full shadow-lg"
            >
              <Calendar className="mr-1 h-4 w-4" />
              Events
            </Button>
            <Button
              size="sm"
              variant={filter === "services" ? "default" : "secondary"}
              onClick={() => setFilter("services")}
              className="rounded-full shadow-lg"
            >
              <Briefcase className="mr-1 h-4 w-4" />
              Services
            </Button>
          </div>
        </div>
      </div>

      {/* Selected Marker Details */}
      {selectedMarker && (
        <div className="absolute bottom-24 left-0 right-0 z-10 px-4">
          <Card className="glass-strong p-4">
            <div className="flex items-start gap-3">
              {selectedMarker.avatarUrl && (
                <Avatar className="h-12 w-12 border-2 border-border">
                  <AvatarImage src={selectedMarker.avatarUrl || "/placeholder.svg"} />
                  <AvatarFallback>{selectedMarker.title[0]}</AvatarFallback>
                </Avatar>
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold truncate">{selectedMarker.title}</p>
                    {selectedMarker.subtitle && (
                      <p className="text-sm text-muted-foreground truncate">{selectedMarker.subtitle}</p>
                    )}
                  </div>
                  <Badge variant="outline" className="capitalize">
                    {selectedMarker.type}
                  </Badge>
                </div>

                {selectedMarker.type === "pet" && (
                  <div className="mt-2 flex gap-2">
                    <Badge variant="secondary" className="text-xs">
                      {selectedMarker.data.distance} mi away
                    </Badge>
                    <Badge variant="secondary" className="text-xs">
                      {selectedMarker.data.compatibility}% match
                    </Badge>
                  </div>
                )}

                {selectedMarker.type === "event" && (
                  <div className="mt-2">
                    <Badge variant="secondary" className="text-xs">
                      {selectedMarker.data.attendees}/{selectedMarker.data.capacity} attending
                    </Badge>
                  </div>
                )}

                {selectedMarker.type === "service" && (
                  <div className="mt-2">
                    <Badge variant="secondary" className="text-xs">
                      {selectedMarker.data.rating} ★ • {selectedMarker.data.reviews} reviews
                    </Badge>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-3 flex gap-2">
              <Button className="flex-1" size="sm">
                View Details
              </Button>
              <Button variant="outline" size="sm" className="bg-transparent">
                Directions
              </Button>
            </div>
          </Card>
        </div>
      )}

      <BottomNav />
    </div>
  )
}
