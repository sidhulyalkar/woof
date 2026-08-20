"use client"

import { Award } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { Badge as BadgeType } from "@/lib/types"

interface BadgeShowcaseProps {
  badges: BadgeType[]
  maxDisplay?: number
}

export function BadgeShowcase({ badges, maxDisplay = 3 }: BadgeShowcaseProps) {
  const displayBadges = badges.slice(0, maxDisplay)
  const remainingCount = badges.length - maxDisplay

  const getRarityColor = (rarity: BadgeType["rarity"]) => {
    switch (rarity) {
      case "legendary":
        return "bg-gradient-to-br from-yellow-400 to-orange-500"
      case "epic":
        return "bg-gradient-to-br from-purple-400 to-pink-500"
      case "rare":
        return "bg-gradient-to-br from-blue-400 to-cyan-500"
      default:
        return "bg-gradient-to-br from-gray-400 to-gray-500"
    }
  }

  return (
    <div className="flex items-center gap-2">
      {displayBadges.map((badge) => (
        <div
          key={badge.id}
          className={`flex h-10 w-10 items-center justify-center rounded-full ${getRarityColor(badge.rarity)}`}
          title={badge.name}
        >
          <Award className="h-5 w-5 text-white" />
        </div>
      ))}
      {remainingCount > 0 && (
        <Badge variant="secondary" className="text-xs">
          +{remainingCount}
        </Badge>
      )}
    </div>
  )
}
