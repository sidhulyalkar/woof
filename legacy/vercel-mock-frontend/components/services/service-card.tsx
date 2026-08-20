"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Star, MapPin, CheckCircle2, MessageCircle } from "lucide-react"
import type { ServiceProvider } from "@/lib/types"
import Link from "next/link"

interface ServiceCardProps {
  provider: ServiceProvider
}

const serviceTypeLabels: Record<string, string> = {
  "dog-walking": "Dog Walking",
  "pet-sitting": "Pet Sitting",
  training: "Training",
  grooming: "Grooming",
}

const serviceTypeColors: Record<string, string> = {
  "dog-walking": "bg-primary/10 text-primary border-primary/20",
  "pet-sitting": "bg-secondary/10 text-secondary border-secondary/20",
  training: "bg-accent/10 text-accent border-accent/20",
  grooming: "bg-muted text-muted-foreground border-border",
}

export function ServiceCard({ provider }: ServiceCardProps) {
  return (
    <Card className="glass p-4 space-y-4">
      {/* Provider Info */}
      <div className="flex items-start gap-3">
        <Avatar className="w-14 h-14 border-2 border-border">
          <AvatarImage src={provider.avatarUrl || "/placeholder.svg"} />
          <AvatarFallback>{provider.name[0]}</AvatarFallback>
        </Avatar>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <h3 className="font-semibold text-lg">{provider.name}</h3>
            {provider.verified && <CheckCircle2 className="w-5 h-5 text-primary shrink-0" />}
          </div>
          <Badge className={`border mb-2 ${serviceTypeColors[provider.serviceType]}`}>
            {serviceTypeLabels[provider.serviceType]}
          </Badge>
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Star className="w-4 h-4 text-primary fill-primary" />
              <span className="font-medium">
                {provider.rating} ({provider.reviewCount})
              </span>
            </div>
            <div className="flex items-center gap-1">
              <MapPin className="w-4 h-4" />
              <span>{provider.distance} mi</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bio */}
      {provider.bio && <p className="text-sm text-muted-foreground line-clamp-2">{provider.bio}</p>}

      {/* Details */}
      <div className="flex items-center justify-between text-sm">
        <div>
          <p className="font-medium">{provider.priceRange}</p>
          <p className="text-xs text-muted-foreground">Price Range</p>
        </div>
        <div className="text-right">
          <p className="font-medium">{provider.availability.length} slots</p>
          <p className="text-xs text-muted-foreground">Available</p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2 pt-2">
        <Button variant="outline" className="flex-1 bg-transparent">
          View Profile
        </Button>
        <Button asChild className="flex-1 gap-2">
          <Link href={`/inbox?provider=${provider.id}`}>
            <MessageCircle className="w-4 h-4" />
            Contact
          </Link>
        </Button>
      </div>
    </Card>
  )
}
