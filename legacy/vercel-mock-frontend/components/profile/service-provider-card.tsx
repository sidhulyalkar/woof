"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Star, Briefcase, DollarSign } from "lucide-react"

interface ServiceProviderCardProps {
  type: string
  rating: number
  reviewCount: number
  priceRange: string
}

const serviceTypeLabels: Record<string, string> = {
  "dog-walking": "Dog Walking",
  "pet-sitting": "Pet Sitting",
  training: "Training",
  grooming: "Grooming",
}

export function ServiceProviderCard({ type, rating, reviewCount, priceRange }: ServiceProviderCardProps) {
  return (
    <Card className="glass p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Briefcase className="w-5 h-5 text-primary" />
          <span className="font-semibold">{serviceTypeLabels[type] || type}</span>
        </div>
        <Badge variant="secondary">Active</Badge>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="flex items-center gap-2">
          <Star className="w-4 h-4 text-primary fill-primary" />
          <div>
            <p className="text-sm font-medium">
              {rating} ({reviewCount})
            </p>
            <p className="text-xs text-muted-foreground">Rating</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-primary" />
          <div>
            <p className="text-sm font-medium">{priceRange}</p>
            <p className="text-xs text-muted-foreground">Price Range</p>
          </div>
        </div>
      </div>
    </Card>
  )
}
