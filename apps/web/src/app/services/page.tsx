"use client"

import { useState } from "react"
import { BottomNav } from "@/components/bottom-nav"
import { ServiceCard } from "@/components/services/service-card"
import { ServiceFilterSheet } from "@/components/services/service-filter-sheet"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Search, SlidersHorizontal } from "lucide-react"
import type { ServiceProvider } from "@/lib/types"

// Mock service providers
const mockProviders: ServiceProvider[] = [
  {
    id: "sp1",
    userId: "u1",
    name: "Sarah Johnson",
    avatarUrl: "/woman-and-loyal-companion.png",
    serviceType: "dog-walking",
    bio: "Professional dog walker with 5+ years experience. Certified in pet first aid.",
    rating: 4.9,
    reviewCount: 48,
    priceRange: "$30-45/hr",
    location: { lat: 37.7749, lng: -122.4194, address: "San Francisco, CA" },
    distance: 0.8,
    availability: ["Weekday mornings", "Weekday afternoons"],
    verified: true,
  },
  {
    id: "sp2",
    userId: "u2",
    name: "Michael Chen",
    avatarUrl: "/man-with-husky.jpg",
    serviceType: "pet-sitting",
    bio: "Experienced pet sitter offering overnight care in your home. Great with all breeds!",
    rating: 4.8,
    reviewCount: 35,
    priceRange: "$50-75/night",
    location: { lat: 37.7849, lng: -122.4094, address: "San Francisco, CA" },
    distance: 1.2,
    availability: ["Weekends", "Weekday evenings"],
    verified: true,
  },
  {
    id: "sp3",
    userId: "u3",
    name: "Emily Rodriguez",
    avatarUrl: "/yoga-instructor.png",
    serviceType: "training",
    bio: "Certified dog trainer specializing in positive reinforcement methods.",
    rating: 5.0,
    reviewCount: 62,
    priceRange: "$60-90/session",
    location: { lat: 37.7649, lng: -122.4294, address: "San Francisco, CA" },
    distance: 1.5,
    availability: ["Weekday afternoons", "Weekends"],
    verified: true,
  },
]

export default function ServicesPage() {
  const [providers] = useState<ServiceProvider[]>(mockProviders)
  const [filterOpen, setFilterOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [serviceType, setServiceType] = useState<string>("all")

  const filteredProviders = providers.filter((provider) => {
    const matchesSearch = provider.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesType = serviceType === "all" || provider.serviceType === serviceType
    return matchesSearch && matchesType
  })

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="px-4 py-4 max-w-lg mx-auto space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Services</h1>
              <p className="text-sm text-muted-foreground">{filteredProviders.length} providers nearby</p>
            </div>
            <Button variant="outline" size="icon" onClick={() => setFilterOpen(true)} className="bg-transparent">
              <SlidersHorizontal className="w-5 h-5" />
            </Button>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search providers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          <Tabs value={serviceType} onValueChange={setServiceType} className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="all" className="text-xs">
                All
              </TabsTrigger>
              <TabsTrigger value="dog-walking" className="text-xs">
                Walking
              </TabsTrigger>
              <TabsTrigger value="pet-sitting" className="text-xs">
                Sitting
              </TabsTrigger>
              <TabsTrigger value="training" className="text-xs">
                Training
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </header>

      {/* Content */}
      <main className="px-4 py-6 max-w-lg mx-auto space-y-4">
        {filteredProviders.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
              <Search className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">No providers found</h3>
            <p className="text-sm text-muted-foreground">Try adjusting your search or filters</p>
          </div>
        ) : (
          filteredProviders.map((provider) => <ServiceCard key={provider.id} provider={provider} />)
        )}
      </main>

      <ServiceFilterSheet open={filterOpen} onOpenChange={setFilterOpen} />
      <BottomNav />
    </div>
  )
}
