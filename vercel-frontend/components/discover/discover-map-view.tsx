"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import {
  MapPin,
  Star,
  Phone,
  Clock,
  DollarSign,
  Dog,
  Scissors,
  Stethoscope,
  Home,
  ShoppingBag,
  Utensils,
  Trees,
  Mountain,
} from "lucide-react"
import type { Service } from "@/lib/types"
import { cn } from "@/lib/utils"

const categoryIcons = {
  walker: Dog,
  grooming: Scissors,
  vet: Stethoscope,
  sitter: Home,
  "food-store": ShoppingBag,
  restaurant: Utensils,
  park: Trees,
  hike: Mountain,
}

const categoryColors = {
  walker: "text-blue-500",
  grooming: "text-pink-500",
  vet: "text-red-500",
  sitter: "text-purple-500",
  "food-store": "text-orange-500",
  restaurant: "text-green-500",
  park: "text-emerald-500",
  hike: "text-teal-500",
}

const mockServices: Service[] = [
  {
    id: "s1",
    name: "Paws & Walk",
    category: "walker",
    location: { lat: 37.7749, lng: -122.4194, address: "123 Market St, San Francisco" },
    rating: 4.8,
    reviews: 156,
    distance: 0.5,
    priceRange: "$$",
    hours: "7 AM - 7 PM",
    phone: "(415) 555-0123",
    imageUrl: "/dog-walker.png",
    description: "Professional dog walking services with experienced handlers",
  },
  {
    id: "s2",
    name: "Pampered Paws Grooming",
    category: "grooming",
    location: { lat: 37.7849, lng: -122.4094, address: "456 Valencia St, San Francisco" },
    rating: 4.9,
    reviews: 203,
    distance: 1.2,
    priceRange: "$$$",
    hours: "9 AM - 6 PM",
    phone: "(415) 555-0456",
    imageUrl: "/dog-grooming-salon.png",
    description: "Full-service grooming salon specializing in all breeds",
  },
  {
    id: "s3",
    name: "Bay Area Pet Hospital",
    category: "vet",
    location: { lat: 37.7649, lng: -122.4294, address: "789 Mission St, San Francisco" },
    rating: 4.7,
    reviews: 312,
    distance: 0.8,
    priceRange: "$$$",
    hours: "24/7 Emergency",
    phone: "(415) 555-0789",
    imageUrl: "/veterinary-clinic-exterior.png",
    description: "24/7 emergency care and routine veterinary services",
  },
  {
    id: "s4",
    name: "Cozy Paws Pet Sitting",
    category: "sitter",
    location: { lat: 37.7549, lng: -122.4394, address: "321 Folsom St, San Francisco" },
    rating: 4.9,
    reviews: 89,
    distance: 1.5,
    priceRange: "$$",
    hours: "By appointment",
    phone: "(415) 555-0321",
    imageUrl: "/pet-sitter.jpg",
    description: "In-home pet sitting and overnight care services",
  },
  {
    id: "s5",
    name: "Bark & Meow Pet Supply",
    category: "food-store",
    location: { lat: 37.7949, lng: -122.3994, address: "654 Castro St, San Francisco" },
    rating: 4.6,
    reviews: 178,
    distance: 2.1,
    priceRange: "$$",
    hours: "9 AM - 8 PM",
    phone: "(415) 555-0654",
    imageUrl: "/vibrant-pet-store.png",
    description: "Premium pet food, treats, and supplies",
  },
  {
    id: "s6",
    name: "The Dog Patch Cafe",
    category: "restaurant",
    location: { lat: 37.7449, lng: -122.4494, address: "987 Potrero Ave, San Francisco" },
    rating: 4.5,
    reviews: 245,
    distance: 1.8,
    priceRange: "$$",
    hours: "8 AM - 9 PM",
    phone: "(415) 555-0987",
    imageUrl: "/dog-friendly-cafe.png",
    description: "Pet-friendly cafe with outdoor seating and dog menu",
  },
  {
    id: "s7",
    name: "Golden Gate Dog Park",
    category: "park",
    location: { lat: 37.7699, lng: -122.4544, address: "Golden Gate Park, San Francisco" },
    rating: 4.8,
    reviews: 567,
    distance: 2.5,
    hours: "6 AM - 10 PM",
    imageUrl: "/lively-dog-park.png",
    description: "Large off-leash dog park with separate areas for small and large dogs",
  },
  {
    id: "s8",
    name: "Lands End Coastal Trail",
    category: "hike",
    location: { lat: 37.7833, lng: -122.5085, address: "Lands End, San Francisco" },
    rating: 4.9,
    reviews: 892,
    distance: 4.2,
    hours: "Sunrise - Sunset",
    imageUrl: "/coastal-hiking-trail.jpg",
    description: "Scenic coastal trail with stunning ocean views, dog-friendly",
  },
]

export function DiscoverMapView() {
  const [selectedService, setSelectedService] = useState<Service | null>(null)
  const [activeFilters, setActiveFilters] = useState<Service["category"][]>([
    "walker",
    "grooming",
    "vet",
    "sitter",
    "food-store",
    "restaurant",
    "park",
    "hike",
  ])

  const filteredServices = mockServices.filter((service) => activeFilters.includes(service.category))

  const toggleFilter = (category: Service["category"]) => {
    setActiveFilters((prev) => (prev.includes(category) ? prev.filter((c) => c !== category) : [...prev, category]))
  }

  return (
    <div className="h-[calc(100vh-185px)]">
      {/* Filter chips */}
      <div className="px-4 py-3 border-b border-border/50 overflow-x-auto">
        <div className="flex gap-2 min-w-max">
          {(Object.keys(categoryIcons) as Service["category"][]).map((category) => {
            const Icon = categoryIcons[category]
            const isActive = activeFilters.includes(category)
            return (
              <Button
                key={category}
                variant={isActive ? "default" : "outline"}
                size="sm"
                onClick={() => toggleFilter(category)}
                className={cn("capitalize shrink-0", !isActive && "bg-transparent")}
              >
                <Icon className="w-4 h-4 mr-1" />
                {category.replace("-", " ")}
              </Button>
            )
          })}
        </div>
      </div>

      {/* Map placeholder */}
      <div className="relative h-[300px] bg-muted border-b border-border/50">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center space-y-2">
            <MapPin className="w-12 h-12 mx-auto text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Interactive map view</p>
            <p className="text-xs text-muted-foreground">Showing {filteredServices.length} nearby services</p>
          </div>
        </div>

        {/* Map markers simulation */}
        {filteredServices.slice(0, 5).map((service, index) => {
          const Icon = categoryIcons[service.category]
          return (
            <button
              key={service.id}
              className={cn(
                "absolute w-10 h-10 rounded-full glass-strong border-2 border-background flex items-center justify-center hover:scale-110 transition-transform",
                categoryColors[service.category],
              )}
              style={{
                left: `${20 + index * 15}%`,
                top: `${30 + (index % 3) * 20}%`,
              }}
              onClick={() => setSelectedService(service)}
            >
              <Icon className="w-5 h-5" />
            </button>
          )
        })}
      </div>

      {/* Services list */}
      <div className="overflow-y-auto h-[calc(100%-300px)] px-4 py-4 space-y-3">
        {filteredServices.map((service) => {
          const Icon = categoryIcons[service.category]
          return (
            <Card
              key={service.id}
              className="p-4 cursor-pointer hover:bg-accent/50 transition-colors"
              onClick={() => setSelectedService(service)}
            >
              <div className="flex gap-3">
                <div
                  className={cn(
                    "w-12 h-12 rounded-lg glass flex items-center justify-center shrink-0",
                    categoryColors[service.category],
                  )}
                >
                  <Icon className="w-6 h-6" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-sm truncate">{service.name}</h3>
                      <Badge variant="secondary" className="text-xs capitalize mt-1">
                        {service.category.replace("-", " ")}
                      </Badge>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="flex items-center gap-1">
                        <Star className="w-4 h-4 fill-yellow-500 text-yellow-500" />
                        <span className="text-sm font-semibold">{service.rating}</span>
                      </div>
                      <p className="text-xs text-muted-foreground">({service.reviews})</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                    <span>{service.distance} mi</span>
                    {service.priceRange && <span>{service.priceRange}</span>}
                    {service.hours && <span>{service.hours}</span>}
                  </div>
                </div>
              </div>
            </Card>
          )
        })}
      </div>

      {/* Service detail sheet */}
      <Sheet open={!!selectedService} onOpenChange={() => setSelectedService(null)}>
        <SheetContent side="bottom" className="h-[85vh]">
          {selectedService && (
            <>
              <SheetHeader>
                <SheetTitle>{selectedService.name}</SheetTitle>
              </SheetHeader>
              <div className="mt-6 space-y-4">
                {/* Image */}
                <div className="aspect-video rounded-lg overflow-hidden bg-muted">
                  <img
                    src={selectedService.imageUrl || "/placeholder.svg"}
                    alt={selectedService.name}
                    className="w-full h-full object-cover"
                  />
                </div>

                {/* Rating & Category */}
                <div className="flex items-center justify-between">
                  <Badge variant="secondary" className="capitalize">
                    {selectedService.category.replace("-", " ")}
                  </Badge>
                  <div className="flex items-center gap-2">
                    <Star className="w-5 h-5 fill-yellow-500 text-yellow-500" />
                    <span className="font-semibold">{selectedService.rating}</span>
                    <span className="text-sm text-muted-foreground">({selectedService.reviews} reviews)</span>
                  </div>
                </div>

                {/* Description */}
                <p className="text-sm text-muted-foreground">{selectedService.description}</p>

                {/* Details */}
                <div className="space-y-3 pt-2">
                  <div className="flex items-start gap-3">
                    <MapPin className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium">Location</p>
                      <p className="text-sm text-muted-foreground">{selectedService.location.address}</p>
                      <p className="text-xs text-muted-foreground mt-1">{selectedService.distance} miles away</p>
                    </div>
                  </div>

                  {selectedService.hours && (
                    <div className="flex items-start gap-3">
                      <Clock className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
                      <div>
                        <p className="text-sm font-medium">Hours</p>
                        <p className="text-sm text-muted-foreground">{selectedService.hours}</p>
                      </div>
                    </div>
                  )}

                  {selectedService.phone && (
                    <div className="flex items-start gap-3">
                      <Phone className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
                      <div>
                        <p className="text-sm font-medium">Phone</p>
                        <p className="text-sm text-muted-foreground">{selectedService.phone}</p>
                      </div>
                    </div>
                  )}

                  {selectedService.priceRange && (
                    <div className="flex items-start gap-3">
                      <DollarSign className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
                      <div>
                        <p className="text-sm font-medium">Price Range</p>
                        <p className="text-sm text-muted-foreground">{selectedService.priceRange}</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-4">
                  <Button className="flex-1">Get Directions</Button>
                  {selectedService.phone && (
                    <Button variant="outline" className="flex-1 bg-transparent">
                      Call Now
                    </Button>
                  )}
                </div>
              </div>
            </>
          )}
        </SheetContent>
      </Sheet>
    </div>
  )
}
